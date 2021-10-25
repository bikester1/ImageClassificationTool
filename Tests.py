from gui import *
from Protocols import *

from unittest import TestCase


class ImageDataTests(TestCase):
    
    image = ImageData("G:\\Hashed Pictures\\fa\\faf9fafb-f6f5f6f7-fbfafbfb-fbfbfbfc-fcfbfcfd.PNG")
    
    def test_defferement(self):
        print(f"{self.image.image}")
        print(f"{self.image.image}")


class LazyInitTests(TestCase):
    
    class TestClassOne:
        
        def __init__(self):
            self.ran = 0
        
        @lazy_init_property
        def test_method_1(self) -> int:
            self.ran += 1
            return 1
            
    def test_init(self):
        obj1 = self.TestClassOne()
        self.assertEqual(0, obj1.ran)
    
    def test_ran_once(self):
        obj1 = self.TestClassOne()
        obj1.test_method_1
        obj1.test_method_1
        obj1.test_method_1
        self.assertEqual(1, obj1.ran)
    
    def test_consitently_one(self):
        obj1 = self.TestClassOne()
        self.assertEqual(1, obj1.test_method_1)
        self.assertEqual(1, obj1.test_method_1)
        self.assertEqual(1, obj1.test_method_1)
        self.assertEqual(1, obj1.test_method_1)
    
    def test_run_set(self):
        obj1 = self.TestClassOne()
        obj1.test_method_1
        obj1.test_method_1 = 2
        self.assertEqual(2, obj1.test_method_1)
    
    def test_run_set_run(self):
        obj1 = self.TestClassOne()
        obj1.test_method_1
        obj1.test_method_1 = 2
        obj1.test_method_1 = None
        self.assertEqual(1, obj1.test_method_1)


class ObservableTests(TestCase):
    
    class TestClass(Observable):
        
        def __init__(self):
            self.count = 0
        
        def call_count(self, var):
            self.count += 1
        
        @updates("count")
        def updates_count(self):
            pass
    
    def setUp(self) -> None:
        self.obj = self.TestClass()
        self.obj.attach(self.obj.call_count, "count")
    
    def test_callback_none(self):
        self.assertEqual(self.obj.count, 0)
    
    def test_callback_1(self):
        self.obj.updates_count()
        self.assertEqual(self.obj.count, 1)
