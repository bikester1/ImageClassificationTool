"""Test cases for project"""
from pathlib import Path
from unittest import TestCase

from data import ImageData
from protocols import LazyInitProperty, Observable, updates


class ImageDataTests(TestCase):
    """Test cases for ImageData"""
    image = ImageData(Path("G:\\Hashed Pictures\\fa\\faf9fafb-f6f5f6f7-fbfafbfb-fbfbfbfc-fcfbfcfd" \
                         ".PNG"))

    def test_defferement(self):
        """ToDo: Implement tests for ImageData"""
        print(f"{self.image.img}")
        print(f"{self.image.img}")


class LazyInitTests(TestCase):
    """Test cases for LazyInitProperties"""
    class TestClassOne:
        """Class used for testing decorator"""
        def __init__(self):
            self.ran = 0

        @LazyInitProperty
        def test_method_1(self) -> int:
            """Ads to run count when executed"""
            self.ran += 1
            return 1

    def test_init(self):
        """Lazy Property will not have been run on init"""
        obj1 = self.TestClassOne()
        self.assertEqual(0, obj1.ran)

    # pylint: disable=pointless-statement
    def test_ran_once(self):
        """Multiple accesses to the property only runs it once"""
        obj1 = self.TestClassOne()
        obj1.test_method_1
        obj1.test_method_1
        obj1.test_method_1
        self.assertEqual(1, obj1.ran)

    def test_consitently_one(self):
        """Value is cached and returned over multiple get calls"""
        obj1 = self.TestClassOne()
        self.assertEqual(1, obj1.test_method_1)
        self.assertEqual(1, obj1.test_method_1)
        self.assertEqual(1, obj1.test_method_1)
        self.assertEqual(1, obj1.test_method_1)

    def test_run_set(self):
        """If property is explicitly set the given value should be cached"""
        obj1 = self.TestClassOne()
        obj1.test_method_1
        obj1.test_method_1 = 2
        self.assertEqual(2, obj1.test_method_1)

    def test_run_set_run(self):
        """If the property is set to None then the computation will be run when it is next
        accessed"""
        obj1 = self.TestClassOne()
        obj1.test_method_1
        obj1.test_method_1 = 2
        obj1.test_method_1 = None
        self.assertEqual(1, obj1.test_method_1)


class ObservableTests(TestCase):
    """Test cases for Observable protocol"""
    class TestClass(Observable):
        """Test class that explicitly implements Observable"""
        def __init__(self):
            super().__init__()
            self.count = 0

        def call_count(self, var):
            """Increased count when called"""
            self.count += 1

        @updates("count")
        def updates_count(self):
            """Calls callbacks when called."""

    def setUp(self) -> None:
        """Pre-Test setup"""
        self.obj = self.TestClass()
        self.obj.attach(self.obj.call_count, "count")

    def test_callback_none(self):
        """Init does not call update to count when set"""
        self.assertEqual(self.obj.count, 0)

    def test_callback_1(self):
        """Callback is called when an updates method is called."""
        self.obj.updates_count()
        self.assertEqual(self.obj.count, 1)
