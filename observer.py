
class Observer:

    def __init__(self, observable=None):
        if observable != None:
            observable.register_observer(self)

    def notify(self, observable, *args, **kwargs):
        print('Got', args, kwargs, 'From', observable)
