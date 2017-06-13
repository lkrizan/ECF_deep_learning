import os
import time

if __name__ == "__main__":
    EXE_PATH = ""
    EXE_NAME = "DeepModelTraining.exe"
    NUM_BATCH = 1
    parameters = {}
    parameters["adam"] = "parameters_adam.txt"
    parameters["SST"] = "parameters_SST.txt"
    parameters["genAnnealing"] = "parameters_genAnnealing.txt"
    parameters["eStrategy"] = "parameters_eStrategy.txt"
    exe_full_path = os.path.join(EXE_PATH, EXE_NAME)
    # disable tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print("Starting batch run....\n")
    for algorithm in parameters.keys():
        # create log file
        log_file_name = "log_" + algorithm + ".txt"
        fp = open(log_file_name, "w")
        fp.close()
        # run algorithm in batch mode
        for i in range(NUM_BATCH):
            # get start time
            print("algorithm: " + algorithm + "batch: " + str(i))
            start = time.time()
            os.system(exe_full_path + " " + parameters[algorithm] + " >> " + log_file_name)
            end = time.time()
            fp = open(log_file_name, "a")
            fp.write("Execution time: " + str(end - start))
            fp.close()
            print("\tExecution time: " + str(end - start))
    print("Batch run finished successfully!")
