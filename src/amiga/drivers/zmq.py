import pickle
import threading
from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol
import time
import zmq

DEFAULT_ROBOT_PORT = 6000
DEFAULT_POLL_FREQUENCY_HZ = 30


class ZMQBackendObject(Protocol):
    @abstractmethod
    def get_methods(self) -> Dict[str, str]:
       """Get the methods that can be called on the robot.
        
        Example:
        obj = AMIGA()
        obj.get_methods() -> {
            "get_num_dofs": None,
            "is_freedrive_enabled": None,
            "servo_joint_positions": ["joint_state"],
            ...
            }
        """
       raise NotImplementedError()

    def make_zmq_client(
        self,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
        async_method: Optional[str] = None,
    ):
        methods = self.get_methods()
        if async_method is not None:
            assert async_method in methods, (
                f"Invalid async_method: {async_method}, available: {list(methods.keys())}"
            )
            return AsyncZMQClient(port, host, async_method, methods)
        return SyncZMQClient(port, host, methods)


class BaseZMQClient:
    """Base class for ZMQ clients."""

    def __init__(self, port: int, host: str):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

        self.host = host
        self.port = port

    def __str__(self):
        return f"{self.__class__.__name__}(host={self.host}, port={self.port})"


class SyncZMQClient(BaseZMQClient):
    """A synchronous ZMQ client."""

    def __init__(self, port: int, host: str, methods: Dict[str, str]):
        super().__init__(port, host)
        self._create_methods(methods)

    def _create_methods(self, methods: Dict[str, str]):
        """Dynamically create synchronous methods."""
        for name, args in methods.items():
            setattr(self, name, self._make_sync_method(name))

    def _make_sync_method(self, name: str):
        """Create a synchronous method."""
        def method(*args, **kwargs):
            if len(args) > 0:
                raise ValueError(f"Positional arguments are not supported: {args}. Use kwargs instead.")
            if hasattr(self, "socket_lock"):
                with self.socket_lock:
                    request = {"method": name, "args": kwargs}
                    self._socket.send(pickle.dumps(request))
                    res = pickle.loads(self._socket.recv())
                    if isinstance(res, dict) and "error" in res:
                        print(f"Error in {name}({kwargs}): {res['error']}")
                    return res
            else:
                request = {"method": name, "args": kwargs}
                self._socket.send(pickle.dumps(request))
                res = pickle.loads(self._socket.recv())
                if isinstance(res, dict) and "error" in res:
                    print(f"Error in {name}({kwargs}): {res['error']}")
                return res
                

        return method


class AsyncZMQClient(SyncZMQClient):
    """Same as SyncZMQClient, but one of the methods will be called at a fixed frequency in the background
    and return the latest result when called to reduce latency.
    All other methods are made available as synchronous methods.
    """

    def __init__(self, port: int, host: str, async_method: str, methods: Dict[str, str] = None):
        methods = {k: v for k, v in methods.items() if k != async_method}
        super().__init__(port, host, methods=methods)
        
        self.async_method = async_method
        self.latest = None
        self.latest_lock = threading.Lock()
        self.socket_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        self._create_dynamic_read_method()

    def _reader(self):
        """Continuously fetch the latest data."""
        while not self.stop_event.is_set():
            request = {"method": self.async_method, "args": {}}
            with self.socket_lock:
                self._socket.send(pickle.dumps(request))
                result = pickle.loads(self._socket.recv())
            with self.latest_lock:
                self.latest = result
            time.sleep(1.0 / DEFAULT_POLL_FREQUENCY_HZ)

    def _create_dynamic_read_method(self):
        """Dynamically creates a read method."""
        def dynamic_read():
            with self.latest_lock:
                return self.latest

        setattr(self, self.async_method, dynamic_read)

    def stop(self):
        """Stop the background reader thread."""
        self.stop_event.set()
        self.thread.join()


class ZMQServer:
    def __init__(self, backend: ZMQBackendObject, host: str, port: int):
        self._backend = backend
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        self._stop_event = threading.Event()

    def serve(self):
        """Serve requests."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # Set timeout to 1000 ms
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)
                method_name = request.get("method")
                args = request.get("args", {})
                method = getattr(self._backend, method_name, None)
                assert method, f"Invalid method: {method_name}"
                result = method(**args)
                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                continue
            except Exception as e:
                print(f"ERROR: {e}")
                self._socket.send(pickle.dumps({"error": str(e)}))

    def stop(self):
        """Stop serving requests."""
        self._stop_event.set()
