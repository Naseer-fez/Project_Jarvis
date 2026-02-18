class HardBlockedInV1Error(Exception):
    pass

class SerialController:
    def send(self, command): raise HardBlockedInV1Error("BLOCKED in V1")
    def move(self, axis, value): raise HardBlockedInV1Error("BLOCKED in V1")
    def connect(self, port): raise HardBlockedInV1Error("BLOCKED in V1")
    def status(self): raise HardBlockedInV1Error("BLOCKED in V1")