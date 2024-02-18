from multiprocessing import Process, Pipe
import gym


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            img = None
            if done:
                obs, human_view = env.reset()
                info = (info[0], info[1], info[2], info[3], human_view)
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs, human_view = env.reset()
            conn.send((obs, human_view))
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        obs, human_view = self.envs[0].reset()  # 20221229 fk: adjusted to contain img
        results = zip(*[(obs, human_view)] + [local.recv() for local in self.locals])  # 20221229 fk: adjusted to contain img
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        img = None
        if done:
            obs, human_view = self.envs[0].reset()
            info = (info[0], info[1], info[2], info[3], human_view)
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()
