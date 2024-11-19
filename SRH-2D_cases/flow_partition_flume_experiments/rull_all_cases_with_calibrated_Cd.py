"""
Run all cases with calibrated Cd. The case is "case_final".
"""

import os
import subprocess
import shutil

if __name__ == "__main__":

    # List of directories
    directories = ['Exp_1_Cd',
                   'Exp_2_Cd',
                   'Exp_3_Cd',
                   'Exp_4_Cd'
                   ]

    script_name = 'run_with_calibrated_Cd.py'

    # Loop over the list of directories
    for directory in directories:
        try:
            # Save the current working directory
            original_dir = os.getcwd()

            # Change the current working directory to the target directory
            os.chdir(directory)
            print(f"Entering directory: {directory}")

            # Copy script file to the current directory
            shutil.copy("../"+script_name, "./")

            # Run the Python script in the current directory using subprocess
            result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)

            # Print the output from the script
            print(f"Output from {script_name} in {directory}:")
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            # Handle errors in the subprocess call (e.g., script not found or runtime error)
            print(f"Error running {script_name} in {directory}: {e.stderr}")

        except Exception as e:
            # Handle general exceptions (e.g., directory change errors)
            print(f"Error in directory {directory}: {e}")

        finally:
            # Always go back to the original working directory
            os.chdir(original_dir)
            print(f"Returning to directory: {original_dir}")

    print("All done!")