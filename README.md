# AMIGA Primitives


## ZMQ communication
We use ZMQ to communicate between modules. All the modules (`AMIGA`, `ZEDCamera`, `RealsenseCamera`) inherit from `amiga.drivers.zmq.ZMQBackend`; this allows the ZMQ server and clients to be generated dynamically from the class itself, using the `get_methods` method. Here how it works:
- Every child of `ZMQBackend` must implement the `get_methods` method, which returns a dictionary with the methods and args that the class wants to expose to the ZMQ server.
- To create a ZMQ server, simply instantiate your class and pass it as an argument to `amiga.driver.zmq.ZMQServer`, which only requires a hostname and port. It will automatically create a server with the methods and args specified in the `get_methods` method.
- For the client side, each backend inherits the `make_zmq_client()` method which creates a `ZMQClient` with dynamically generated methods to match the backend.

Here is an example:

```python
import amiga

class MyNewBackend(amiga.drivers.zmq.ZMQBackend):
    def get_methods(self):
        return {'my_method': ['arg1', 'arg2']}

    def my_method(self, arg1, arg2):
        return arg1 + arg2


### Server Side

# This is a yaml file as described in cfg/zmq_backend.yaml
# It has the hostname, port
cfg = omegaconf.OmegaConf.load("path") 
back = MyNewBackend()
server = amiga.drivers.zmq.ZMQServer(cfg.hostname, cfg.port, back)
server.serve()

### Client side
cfg = omegaconf.OmegaConf.load("path")  # Reuse the same file
back = MyNewBackend()
client = back.make_zmq_client(cfg.hostname, cfg.port)
client.my_method(1, 2)  # Returns 3
```