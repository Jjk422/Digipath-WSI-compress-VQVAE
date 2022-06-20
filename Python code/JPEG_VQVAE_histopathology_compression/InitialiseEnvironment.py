def Initialise(
	conda_requirements_directory_path=None,
	create_files=True,
	copy_library_code=True,
	library_code_location="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/new_code_rewrite",
	directory_to_copy_library_code_to="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history",
	main_filename=None,
	network_origin_file_path="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/networks",
	network_copy_file_dir="/nobackup/projects/bdlds05/jasonk/JPEG_VQVAE_histopathology_compression/experiment_history",
	network_file_name="VQVAE.py"
):
    # Add libraries path to system path so python can easily access it
    import os, sys, shutil

    print("--- Initialising activating library into python path ---")
    currentdir = os.getcwd()
    parentdir = os.path.dirname(currentdir)
    sys.path.append(f"{parentdir}/new_code_rewrite")
    print(f"Current directory: {currentdir}")
    print(f"Parent directory: {parentdir}")
    print("Added library path to python system path")
    print(f"Library directory: {parentdir}/new_code_rewrite")

    from datetime import datetime
    print("--- Initialising datetime functions ---")
    date_time = datetime.today()
    date_today = date_time.strftime("%Y-%m-%d")
    date_time_today = date_time.today().strftime("%Y_%m_%d-%H_%M_%S")

    print(f"Init function run at {datetime.today()}")
    print(f"Main datetime given as: {date_time_today}")

    if create_files:
        # Save conda environment to requirements file
        print("--- Saving conda requirements file ---")
        conda_requirements_directory_path = "conda_requirements_files" if conda_requirements_directory_path is None else conda_requirements_directory_path
        conda_requirements_filename = f"requirements_{date_time_today}.txt"
        conda_requirements_file_path = f"{conda_requirements_directory_path}/{date_today}/{date_time_today}/{conda_requirements_filename}"

        print(f"Conda requirements directory path: {conda_requirements_directory_path}")
        print(f"Conda requirements filename: {conda_requirements_filename}")
        print(f"Conda requirements file path: {conda_requirements_file_path}")
        os.makedirs(conda_requirements_directory_path, exist_ok=True)
        os.system(f"conda list -e > {conda_requirements_file_path}")

        network_file_origin_dir = "".join(network_origin_file_path.split(network_file_name)).replace("\\", "/")
        network_file_new_dir = f"{network_copy_file_dir}/{date_today}/{date_time_today}"
        print(f"Model file origin directory: {network_file_origin_dir}")
        print(f"Model file filename: {network_file_name}")
        print(f"Model file copy path: {network_file_new_dir}/{network_file_name}")

        os.makedirs(f"{network_file_new_dir}", exist_ok=True)
        shutil.copyfile(f"{network_file_origin_dir}/{network_file_name}", f"{network_file_new_dir}/{network_file_name}")

        if copy_library_code:
            if directory_to_copy_library_code_to is None:
                print("directory_to_copy_library_code_to argument cannot be none, specify the root directory for the library code to be copied to")
                exit(0)
            print(f"Copying current library from {parentdir}/new_code_rewrite to {directory_to_copy_library_code_to}/{date_today}/{date_time_today}/Library_code/new_code_rewrite")
            shutil.copytree(f"{library_code_location}/new_code_rewrite", f"{directory_to_copy_library_code_to}/{date_today}/{date_time_today}/Library_code/new_code_rewrite")

        init_script_filename = __file__.split('/')[-1].split('\\')[-1]
        main_script_filename = main_filename.split('/')[-1].split('\\')[-1]

        main_filenames = [
            (init_script_filename, "Init_file"),
            (main_script_filename, "Main_script_run"),
        ]

        for (filename, script_type) in main_filenames:
            print(f"{script_type} {filename} copied to: {directory_to_copy_library_code_to}/{date_today}/{date_time_today}/{script_type}---{filename}")
            os.makedirs(f"{directory_to_copy_library_code_to}/{date_today}/{date_time_today}", exist_ok=True)
            shutil.copy(filename, f"{directory_to_copy_library_code_to}/{date_today}/{date_time_today}/{script_type}---{filename}")

    print("--- Initialisation completed ---")
    print()

    return currentdir, parentdir, date_today, date_time_today, date_time
