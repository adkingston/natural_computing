package main

import (
	"flag"
	"time"

	"github.com/pieterclaerhout/go-log"
)

const (
	linear    = "linear"
	nonlinear = "nonlinear"

	pso = "pso"
	sgd = "sgd"
)

func main() {
	log.DebugMode = true
	log.PrintColors = true
	log.PrintTimestamp = true
	log.TimeFormat = time.RFC3339

	log.Info("Initializing concurrent cross validation")

	var numFolds int
	var optimizer string
	var inputType string

	flag.IntVar(&numFolds, "folds", 10, "the number of folds in the cross validation")
	flag.StringVar(&optimizer, "optimizer", "", "PSO or SGD. Errors if input is not one of these")
	flag.StringVar(&inputType, "input-type", "", "linear or non-linear. Errors if input is not one of these")

	flag.Parse()

	if len(optimizer) > 0 && optimizer != pso && optimizer != sgd {
		log.Errorf("invalid optimizer. Choose either PSO or SGD. Optimizer provided:%s", optimizer)
		return
	}

	if len(inputType) > 0 && inputType != linear && inputType != nonlinear {
		log.Errorf("invalid input type. Choose either linear or nonlinear")
		return
	}

	log.Infof("parameters:: k: %d, opt: %s, inputType: %s", numFolds, optimizer, inputType)

	// code for cross validation
	err := prepareValidation(numFolds)
	if err != nil {
		log.Error(err)
		return
	}

	validations := []*crossValidation{}

	opts := []string{}
	if len(optimizer) == 0 {
		opts = []string{"pso", "sgd"}
	}

	inps := []string{}
	if len(inputType) == 0 {
		inps = []string{"linear", "nonlinear"}
	}

	for _, opt := range opts {
		for _, inp := range inps {
			validations = append(validations, &crossValidation{
				Iterations: numFolds,
				Optimizer:  opt,
				InputType:  inp,
			})
		}
	}

	output := make([]*crossValidation, len(validations))
	log.DebugDump(validations, "validators to run")
	for i, v := range validations {
		log.DebugDump(v, "running cross validator")
		// ranging over slice of pointers. need to dereference as the element v
		// is the same address throughout the loop.
		validation := *v
		val := &validation
		val.Run()

		output[i] = val
	}

	err = writeResultsToFile(output, "output.json")
	if err != nil {
		log.Errorf("failed to write output to file. cause: %s", err)
		return
	}
}
