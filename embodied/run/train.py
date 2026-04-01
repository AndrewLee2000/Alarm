import collections
from functools import partial as bind
import jax
import elements
import embodied
import numpy as np


def train(make_agent, make_replay, make_detector_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  detector_replay = make_detector_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(detector_replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_detector_train = iter(agent.stream(make_stream(detector_replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))
  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length: # TODO : wake가 True 되면 이거 처럼 걍 return 시켜서 일정 기간 동안은 data만 모으게 할까
      return
    for _ in range(should_train(step)): # env 1step당 parameter update 횟수
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
        detector_batch = next(stream_detector_train)
      if (agent.model.wake_length[worker] == agent.model.config.wake_length) : 
        import ast # HACK # TODO : DT가 판단하는 코드 들어가야 함 - agent.model.wake[worker] = detector(detector_batch) : 65step에 대해서 16개의 궤적이 있으므로 총 16개의 환경 판단이 이뤄 져야 함.
        agent.model.wake[worker] = ast.literal_eval(input(f'wake (True/False){worker}: ')) # HACK 
      if agent.model.wake[worker] and agent.model.wake_length[worker] > 0:
        carry_train[0], outs, mets = agent.wake_train(carry_train[0], detector_batch)
        agent.model.wake_length[worker] -= 1
        jax.debug.print(f"✅ wake_length(wake_train){worker}: {agent.model.wake_length[worker]}") # HACK
        if agent.model.wake_length[worker] == 0:
          agent.model.wake[worker] = False
          agent.model.wake_length[worker] = agent.model.config.wake_length
      else:
        carry_train[0], outs, mets = agent.dream_train(carry_train[0], batch)
        jax.debug.print(f"✅ wake_length(dream_train){worker}: {agent.model.wake_length[worker]}") # HACK
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10) # TODO : train_ratio를 조절 해야 하지 않을까..., 환경 변화 판단 코드 들어가야...

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()
