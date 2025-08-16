import sys

class OutputTee:
    """Tees standard output to both the terminal and a file."""
    def __init__(self, filename):
        self.terminal = sys.__stdout__  # Use sys.__stdout__ for the original
        self.file = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

def capture_and_store_output(filename="terminal_output.txt"):
    """Captures standard output and writes it to both the terminal and a file."""
    sys.stdout = OutputTee(filename)
    print("Starting output capture...")

def stop_capture_and_restore_output():
    """Restores standard output and closes the output file."""
    if isinstance(sys.stdout, OutputTee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print("\nOutput capture stopped and restored.")
    else:
        print("\nOutput capture was not active.")