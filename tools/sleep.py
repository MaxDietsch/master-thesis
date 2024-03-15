import time

def wait_and_print():
  """Waits for 5 minutes and then prints a message."""
  print("Five minutes have passed!")
  time.sleep(5 * 60)  # Convert 5 minutes to seconds

if __name__ == "__main__":
    while True:
        wait_and_print()


