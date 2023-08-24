class PrintSection:
    def __init__(self, section_name: str):
        self.name = section_name

    def __enter__(self):
        print()
        print(self.name.center(50, '>'))

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.name.center(50, '<'))
        if exc_type == KeyboardInterrupt:
            print('Interrupted by Ctrl+C')
        else:
            print()
