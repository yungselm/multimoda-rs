# tests/test_wrappers.py
import pytest
from multimodars._wrappers import from_file, from_array, to_centerline
from multimodars import PyGeometry, PyGeometryPair, PyCenterline

def test_from_file_single(mocker, tmp_path):
    mock_fn = mocker.patch('multimodars._wrappers.from_file_single')
    from_file("single", input_path=str(tmp_path / "test.csv"))
    mock_fn.assert_called_once()

def test_from_array_single(sample_geometry):
    result = from_array("single", geometry=sample_geometry)
    assert isinstance(result, PyGeometry)

def test_to_centerline_three_pt(sample_geometry):
    cl = PyCenterline(points=[])
    geom_pair = PyGeometryPair(sample_geometry, sample_geometry)
    result = to_centerline(
        "three_pt",
        centerline=cl,
        geometry_pair=geom_pair,
        aortic_ref_pt=(0,0,0),
        upper_ref_pt=(1,1,1),
        lower_ref_pt=(2,2,2)
    )
    assert isinstance(result, PyGeometryPair)