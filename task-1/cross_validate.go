package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"time"

	"github.com/pieterclaerhout/go-log"
)

const (
	iterations = 5
)

// this struct stores the results of a training run
type nnResults struct {
	iteration    int       // for local use
	TrainingLoss float64   `json:"training_loss,omitempty"`
	TestingLoss  float64   `json:"testing_loss,omitempty"`
	BestWeight   []float64 `json:"best_weight,omitempty"`
}

type crossValidationOutput struct {
	AverageTrainingLoss float64 `json:"average_training_loss,omitempty"`
	AverageTestingLoss  float64 `json:"average_testing_loss,omitempty"`
	// means the weight from the best performing iteration
	BestWeight    []float64             `json:"best_weight,omitempty"`
	BestIteration int                   `json:"best_iteration,omitempty"`
	AllResults    [iterations]nnResults `json:"all_results,omitempty"`
}

// partitions the data set in the given file into n pieces.
// each element (line of the original file) is a string, and each partition
// is a list of strings. The final result then, is a list of a list of strings
func partitionDataset(filename string, numPartitions int) ([iterations][]string, error) {
	retval := [iterations][]string{}
	file, err := os.Open(filename)
	if err != nil {
		return retval, err
	}

	defer file.Close()

	buf := bufio.NewReader(file)

	iter := 0
	for {
		line, _, err := buf.ReadLine()
		retval[iter] = append(retval[iter], string(line[:]))

		iter++
		if iter == iterations {
			iter = 0
		}

		switch {
		case err == io.EOF:
			return retval, nil
		case err != nil:
			return retval, err
		}
	}

}

// saves each iterations dataset in the dir data/0{iteration number}/{train/test}.dat
func saveCrossValidationSets(partitions [iterations][]string) error {
	for iter := 0; iter < iterations; iter++ {
		path := "data/" + strconv.Itoa(iter) + "_iter"
		err := os.MkdirAll(path, 0775)
		if err != nil && err.Error() != "mkdir "+path+": file exists" {
			fmt.Println(err.Error())
			return err
		}

		test, err := os.OpenFile(path+"/test.dat", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		defer test.Close()

		testWriter := bufio.NewWriter(test)
		for _, data := range partitions[iter] {
			if len(data) > 0 {
				_, err := testWriter.WriteString(data + "\n")
				if err != nil {
					return err
				}
			}
		}
		testWriter.Flush()

		train, err := os.OpenFile(path+"/train.dat", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		defer train.Close()

		trainWriter := bufio.NewWriter(train)
		for i := 0; i < iterations; i++ {
			if i == iter {
				continue
			}

			for _, data := range partitions[i] {
				if len(data) > 0 {
					_, err = trainWriter.WriteString(data + "\n")
					if err != nil {
						return err
					}
				}
			}
		}
		trainWriter.Flush()

	}
	return nil
}

// calls the python script, parses the results, and sends them on the channel
func trainNeuralNetwork(iteration int, outputChannel chan nnResults) {
	log.Infof("begin iteration %d", iteration)
	res := nnResults{}
	path := "data/" + strconv.Itoa(iteration) + "_iter"
	py := exec.Command("python", "pso_pytorch.py", path)

	var out bytes.Buffer
	var stdErr bytes.Buffer

	py.Stdout = &out
	py.Stderr = &stdErr

	startTime := time.Now()
	err := py.Run()
	if err != nil {
		log.Errorf("python script failed: %s", stdErr.String())
		outputChannel <- res
		return
	}

	err = json.Unmarshal(out.Bytes(), &res)
	if err != nil {
		log.Errorf("failed to unmarshal output: %s", err)
		log.ErrorDump(out.String(), "failed output")
		outputChannel <- res
		return
	}
	res.iteration = iteration
	outputChannel <- res
	log.Infof("end iteration %d. testing loss: %.5f, duration: %.5f seconds", iteration, res.TestingLoss, time.Since(startTime).Seconds())
}

// saves final result to file and prints them out as well
func saveResults(results [iterations]nnResults) {
	retval := crossValidationOutput{}

	retval.AllResults = results
	trainingAverage := 0.0
	testingAverage := 0.0

	bestLoss := 1.0
	for _, res := range results {
		trainingAverage += res.TrainingLoss
		testingAverage += res.TestingLoss

		if res.TestingLoss < bestLoss {
			retval.BestWeight = res.BestWeight
			retval.BestIteration = res.iteration
		}
	}

	retval.AverageTrainingLoss = trainingAverage / float64(len(results))
	retval.AverageTestingLoss = testingAverage / float64(len(results))

	//log.InfoDump(retval, "")

	// write to file
	err := writeResultsToFile(retval, "output.json")
	if err != nil {
		log.Errorf("failed to write results to file. cause: %v", err)
	}
}

func writeResultsToFile(results crossValidationOutput, filename string) error {

	out, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	writer := bufio.NewWriter(out)
	defer writer.Flush()

	outB, _ := json.MarshalIndent(results, "", "  ")
	_, err = writer.Write(outB)
	if err != nil {
		return err
	}
	return nil
}

func main() {
	startTime := time.Now()
	// initialize logger
	log.PrintColors = true
	log.PrintTimestamp = true
	log.TimeFormat = time.RFC3339

	log.Info("Initializing concurrent cross validation")

	// partition dataset
	partitions, err := partitionDataset("two_spirals.dat", iterations)
	if err != nil {
		log.Errorf("failed to partition dataset. cause: %v", err)
		return
	}

	err = saveCrossValidationSets(partitions)
	if err != nil {
		log.Errorf("failed to save train and test files for each iteration. cause: %v", err)
		return
	}

	// need to run each validation and save the results
	resultsChannel := make(chan nnResults, iterations)
	for i := 0; i < iterations; i++ {
		// concurrently train through all iterations
		go trainNeuralNetwork(i, resultsChannel)
	}

	allResults := [iterations]nnResults{}
	count := 0
	for {
		select {
		case res := <-resultsChannel:
			allResults[res.iteration] = res
			count++

			if count == iterations {
				saveResults(allResults)
				log.Infof("validation took %v", time.Since(startTime))
				return
			}

		}
	}

}
