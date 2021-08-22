r"""Setup fixture for testing :py:mod:`lmp.dset._ch_poem.ChPoemDset`."""

import os

import pytest

from lmp import path
from lmp.dset._ch_poem import ChPoemDset


@pytest.fixture(
    params=[
        '元', '元末明初', '先秦', '南北朝', '唐', '唐末宋初', '宋', '宋末元初', '宋末金初', '明',
        '明末清初', '民國末當代初', '清', '清末民國初', '清末近現代初', '漢', '當代', '秦', '近現代',
        '近現代末當代初', '遼', '金', '金末元初', '隋', '隋末唐初', '魏晉', '魏晉末南北朝初',
    ],
)
def dset_ver(request):
    """Version of dataset"""

    return request.param


@pytest.fixture
def download_dset(dset_ver):
    r"""Download and return ChPoemDset in the function scope"""
    ch_dset = ChPoemDset(ver=dset_ver)

    return ch_dset


@pytest.fixture
def cleandir(dset_ver, download_dset, request):
    r"""Clean the downloaded dataset in the middle of testing"""

    def remove():
        file_path = os.path.join(
            path.DATA_PATH,
            download_dset.file_name.format(dset_ver),
        )

        if os.path.exists(file_path):
            os.remove(file_path)

    request.addfinalizer(remove)


@pytest.fixture(scope="session")
def lastcleandir(request):
    r"""Clean the downloaded dataset at the end of testing session"""

    def remove():
        dset_list = [
            '元', '元末明初', '先秦', '南北朝', '唐', '唐末宋初', '宋', '宋末元初', '宋末金初', '明',
            '明末清初', '民國末當代初', '清', '清末民國初', '清末近現代初', '漢', '當代', '秦', '近現代',
            '近現代末當代初', '遼', '金', '金末元初', '隋', '隋末唐初', '魏晉', '魏晉末南北朝初',
        ]

        for i in dset_list:
            file_path = os.path.join(
                path.DATA_PATH,
                ChPoemDset(ver=i).file_name.format(i),
            )

            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(path.DATA_PATH):
            os.removedirs(path.DATA_PATH)

    request.addfinalizer(remove)
