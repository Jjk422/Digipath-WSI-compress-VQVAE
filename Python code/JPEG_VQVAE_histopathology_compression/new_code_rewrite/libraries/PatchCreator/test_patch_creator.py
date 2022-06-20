import os
from unittest import TestCase
from libraries.PatchCreator.patch_creator import PatchCreator


class TestPatchCreator(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
        # os.remove('some_file_to_test')

    def test_set_metadata(self):
        test_obj = PatchCreator()

        key = "metadata_key"
        value = "metadata_value"

        test_obj.set_metadata(key, value)

        self.assertEqual(test_obj.metadata[key], value)

    def test_set_dataset_name(self):
        test_obj = PatchCreator()

        dataset_name = "test_name"

        test_obj.set_dataset_name(dataset_name)
        self.assertEqual(test_obj.metadata["dataset_name"], dataset_name)

    def test_set_output_patch_root_directory(self):
        test_obj = PatchCreator()

        output_patch_root_directory_name = "test_name"

        test_obj.set_output_patch_root_directory(output_patch_root_directory_name)

        self.assertEqual(test_obj.metadata["output_patch_root_directory"], output_patch_root_directory_name)

    def test_get_output_patch_root_directory(self):
        test_obj = PatchCreator()

        output_patch_root_directory_name = "test_name"

        test_obj.set_output_patch_root_directory(output_patch_root_directory_name)

        self.assertEqual(test_obj.get_output_patch_root_directory(), output_patch_root_directory_name)

    def test_create_output_patch_directory_structure(self):
        test_obj = PatchCreator()

        output_patch_root_directory = "Test_dir_to_del_001357982"

        test_obj.create_output_patch_directory_structure(output_patch_root_directory)

        assert os.path.exists(
            f"{test_obj.metadata['output_patch_root_directory']}/{test_obj.metadata['dataset_name']}/{test_obj.patch_features['tile_size']}") is True

        os.rmdir(
            f"{test_obj.metadata['output_patch_root_directory']}/{test_obj.metadata['dataset_name']}/{test_obj.patch_features['tile_size']}")
        os.rmdir(f"{test_obj.metadata['output_patch_root_directory']}/{test_obj.metadata['dataset_name']}")
        os.rmdir(f"{test_obj.metadata['output_patch_root_directory']}")

    # def test_generate_patches_from_slide(self):
    #
    #     self.fail()
