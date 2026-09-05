"""BusAgent facade; no dependence on Florence, Isaac, or a particular arm."""
class FinePlaceSkill:
    name = "fine_place"
    aliases = ("general_place", "place")

    def __init__(self, node):
        self.node = node

    def execute(self, request, held_object):
        return self.node.execute(request,held_object)

    def cancel(self):
        self.node.cancel()
