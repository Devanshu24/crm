from crm.utils import load_object, save_object
import os

def test_save_object():
	d = {i: i*i for i in range(10)}
	save_object(d, "./test_d.dill")

def test_load_object():
	d = load_object("./test_d.dill")
	d_real = {i: i*i for i in range(10)}
	assert d == d_real
	os.remove("./test_d.dill")