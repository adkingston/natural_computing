package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"time"

	"github.com/pieterclaerhout/go-log"
)

// this struct stores the results of a training run
type nnResults struct {
	Iteration    int      `json:"iteration,omitempty"`
	TrainingLoss float64  `json:"training_loss,omitempty"`
	TestingLoss  float64  `json:"testing_loss,omitempty"`
	TrainEpochs  []string `json:"train_epochs,omitempty"`
	TestEpochs   []string `json:"test_epochs,omitempty"`
	Fitness      float64  `json:"fitness,omitempty"`
}

type crossValidation struct {
	Iterations          int     `json:"iterations,omitempty"`
	Optimizer           string  `json:"optimizer,omitempty"`
	InputType           string  `json:"input_type,omitempty"`
	AverageTrainingLoss float64 `json:"average_training_loss,omitempty"`
	AverageTestingLoss  float64 `json:"average_testing_loss,omitempty"`
	AverageFitness      float64 `json:"average_fitness,omitempty"`
	BestFitness         float64 `json:"best_fitness,omitempty"`
	// means the weight from the best performing iteration
	BestIteration int `json:"best_iteration,omitempty"`
}

type experiment struct {
	LinearPSO    *crossValidation `json:"linear_pso,omitempty"`
	NonlinearPSO *crossValidation `json:"nonlinear_pso,omitempty"`
	LinearSGD    *crossValidation `json:"linear_sgd,omitempty"`
	NonlinearSGD *crossValidation `json:"nonlinear_sgd,omitempty"`
}

func fitness(train_loss, test_loss float64) float64 {
	return math.Abs(train_loss-test_loss) + (train_loss+test_loss)/2.0
}

// partitions the data set in the given file into n pieces.
// each element (line of the original file) is a string, and each partition
// is a list of strings. The final result then, is a list of a list of strings
func partitionDataset(filename string, numPartitions int) ([][]string, error) {
	retval := make([][]string, numPartitions)

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
		if iter == numPartitions {
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
func saveCrossValidationSets(partitions [][]string) error {
	csvPath := "data/figures"
	err := os.MkdirAll(csvPath, 0775)
	if err != nil && err.Error() != "mkdir "+csvPath+": file exists" {
		return err
	}
	for iter := 0; iter < len(partitions); iter++ {
		path := "data/" + strconv.Itoa(iter) + "_iter"
		err := os.MkdirAll(path, 0775)
		if err != nil && err.Error() != "mkdir "+path+": file exists" {
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
		for i := 0; i < len(partitions); i++ {
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
func trainNeuralNetwork(iteration int, optimizer, inputType string, outputChannel chan nnResults) {
	log.Infof("begin iteration %d", iteration)
	res := nnResults{}
	path := "data/" + strconv.Itoa(iteration) + "_iter"

	py := exec.Command("python", "nn_"+optimizer+".py", path, inputType)

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
	res.Fitness = fitness(res.TrainingLoss, res.TestingLoss)
	res.Iteration = iteration
	outputChannel <- res
	log.Infof("end iteration %d. testing loss: %.5f, duration: %.5f seconds", iteration, res.TestingLoss, time.Since(startTime).Seconds())
}

// saves final result to file and prints them out as well
func (c *crossValidation) saveResults(results []nnResults) {
	log.Debug("saving results")
	baseFileName := c.Optimizer + "_" + c.InputType

	trainEpochsCSV, _ := os.OpenFile("data/figures/"+baseFileName+"_train.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	defer trainEpochsCSV.Close()
	testEpochsCSV, _ := os.OpenFile("data/figures/"+baseFileName+"_test.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	defer testEpochsCSV.Close()

	trainWriter := csv.NewWriter(trainEpochsCSV)
	defer trainWriter.Flush()
	testWriter := csv.NewWriter(testEpochsCSV)
	defer testWriter.Flush()

	trainingAverage := 0.0
	testingAverage := 0.0
	fitnessAverage := 0.0

	bestLoss := 1.0
	for _, res := range results {
		trainWriter.Write(res.TrainEpochs)
		testWriter.Write(res.TestEpochs)

		trainingAverage += res.TrainingLoss
		testingAverage += res.TestingLoss
		fitnessAverage += res.Fitness
		if res.TestingLoss < bestLoss {
			c.BestIteration = res.Iteration
		}
	}

	c.AverageTrainingLoss = trainingAverage / float64(c.Iterations)
	c.AverageTestingLoss = testingAverage / float64(c.Iterations)
	c.AverageFitness = fitnessAverage / float64(c.Iterations)
	//log.InfoDump(retval, "")

	// write to file
	//err := writeResultsToFile(retval, baseFileName+"_output.json")
	//if err != nil {
	//log.Errorf("failed to write results to file. cause: %v", err)
	//}
}

func writeResultsToFile(results interface{}, filename string) error {
	log.Debug("writing output file")
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

func prepareValidation(iterations int) error {
	// partition dataset
	// clear out previous
	os.RemoveAll("./data")
	os.Remove("output.json")
	files, err := filepath.Glob("./data/figures/*.csv")
	if err != nil {
		return err
	}

	for _, f := range files {
		os.Remove(f)
	}

	partitions, err := partitionDataset("two_spirals.dat", iterations)
	if err != nil {
		log.Errorf("failed to partition dataset. cause: %v", err)
		return err
	}

	err = saveCrossValidationSets(partitions)
	if err != nil {
		log.Errorf("failed to save train and test files for each iteration. cause: %v", err)
		return err
	}
	return nil
}

func (c *crossValidation) Run() {
	startTime := time.Now()
	// need to run each validation and save the results
	resultsChannel := make(chan nnResults, c.Iterations)
	for i := 0; i < c.Iterations; i++ {
		// concurrently train through all iterations
		go trainNeuralNetwork(i, c.Optimizer, c.InputType, resultsChannel)
	}

	allResults := make([]nnResults, c.Iterations)
	count := 0
	for {
		select {
		case res := <-resultsChannel:
			allResults[res.Iteration] = res
			count++

			if count == c.Iterations {
				c.saveResults(allResults)
				log.Infof("validation took %v", time.Since(startTime))
				return
			}

		}
	}

}
