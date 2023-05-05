#!/usr/bin/env python3

import time

class MyTime(object):
    def __init__(self) -> None:
        pass

    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        curr = time.time()
        print("elapsed time: ", curr - self.start)
        self.start = None