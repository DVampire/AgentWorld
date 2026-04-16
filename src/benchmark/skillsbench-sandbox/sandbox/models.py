import threading
import time


class TaskInfo:
    def __init__(
        self,
        task_path,
        task_id,
        dataset,
        environment_path,
        tests_path,
        instruction_path,
        allow_internet,
        agent_timeout_sec,
        verifier_timeout_sec,
        has_artifact,
    ):
        self.task_path = task_path
        self.task_id = task_id
        self.dataset = dataset
        self.environment_path = environment_path
        self.tests_path = tests_path
        self.instruction_path = instruction_path
        self.allow_internet = allow_internet
        self.agent_timeout_sec = agent_timeout_sec
        self.verifier_timeout_sec = verifier_timeout_sec
        self.has_artifact = has_artifact


class SandboxEnv:
    def __init__(
        self,
        env_id,
        task_info,
        image_tag,
        container_name,
        run_dir,
        logs_dir,
    ):
        self.env_id = env_id
        self.task_info = task_info
        self.image_tag = image_tag
        self.container_name = container_name
        self.run_dir = run_dir
        self.logs_dir = logs_dir
        self.created_at = time.time()
        self.steps = 0
        self.last_reward = None
        self.lock = threading.Lock()

    @property
    def instruction_path(self):
        return self.task_info.instruction_path
