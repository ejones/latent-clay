import json

class WorldLog:
    def __init__(self, path):
        try:
            with open(path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            lines = []
        self._items = [json.loads(line) for line in lines]
        self._path = path
        self.objects = {}
        self.objects_unmasked = {}
        for item in self._items:
            self._handle(*item)

    def _handle(self, id, op, typ, data):
        if op == 'set':
            self.objects[id] = (typ, data)
            if 'mask' not in data:
                self.objects_unmasked[id] = (typ, data)
        else:
            assert op == 'del'
            del self.objects[id]
            if id in self.objects_unmasked:
                del self.objects_unmasked[id]

    def __getitem__(self, idx):
        return self._items[idx]

    def append(self, item):
        with open(self._path, 'a') as f:
            f.write(f'{json.dumps(item)}\n')
        self._items.append(item)
        self._handle(*item)

    def extend(self, items):
        with open(self._path, 'a') as f:
            f.write('\n'.join((json.dumps(item) for item in items)) + '\n')
        self._items.extend(items)
        for item in items:
            self._handle(*item)
