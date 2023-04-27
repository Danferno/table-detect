class StructureArgs(object):
    def __init__(self, out_dir, visualize=True):
        self.visualize = visualize
        self.verbose = False
        self.out_dir = str(out_dir)