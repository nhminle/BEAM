package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"io"
	"path/filepath"
	"strings"
	"sync"
)

// Struct to represent a JSONL record
type Record struct {
	Text string `json:"text"`
}

// Build a hash set from the JSONL file
func buildTextSet(jsonlPath string) (map[string]struct{}, error) {
    textSet := make(map[string]struct{})

    file, err := os.Open(jsonlPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open JSONL file: %w", err)
    }
    defer file.Close()

    reader := bufio.NewReader(file)
    for {
        line, err := reader.ReadString('\n') // Read until the next newline character
        if err != nil {
            if err == io.EOF {
                break
            }
            return nil, fmt.Errorf("failed to read JSONL file: %w", err)
        }
        line = strings.TrimSpace(line) // Remove any trailing newline or whitespace
        var record Record
        if err := json.Unmarshal([]byte(line), &record); err == nil && record.Text != "" {
            textSet[record.Text] = struct{}{}
        }
    }
    return textSet, nil
}

// Search for matches in a single CSV file
func searchInCSV(csvPath string, textColumns []string, textSet map[string]struct{}) ([]string, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open CSV file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV headers: %w", err)
	}

	// Map column names to their indices
	columnIndices := make(map[string]int)
	for i, header := range headers {
		columnIndices[header] = i
	}

	// Check if the required columns exist
	var indices []int
	for _, col := range textColumns {
		if idx, found := columnIndices[col]; found {
			indices = append(indices, idx)
		}
	}
	if len(indices) == 0 {
		return nil, nil // No matching columns
	}

	// Search for matches
	var matches []string
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		for _, idx := range indices {
			if idx < len(record) {
				text := record[idx]
				if _, found := textSet[text]; found {
					fmt.Printf("Found a match : %s",text)
					matches = append(matches, fmt.Sprintf("Text: %s, File: %s", text, csvPath))
				}
			}
		}
	}
	return matches, nil
}

// Save results to an output file
func saveResults(outputPath string, results []string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create results file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	for _, line := range results {
		if _, err := writer.WriteString(line + "\n"); err != nil {
			return fmt.Errorf("failed to write to results file: %w", err)
		}
	}
	return nil
}

// Recursively find all CSV files in a directory
func findCSVFiles(baseDir string) ([]string, error) {
	var csvFiles []string
	err := filepath.Walk(baseDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".csv") {
			csvFiles = append(csvFiles, path)
		}
		return nil
	})
	return csvFiles, err
}

// Process CSV files in parallel
func processCSVFiles(csvFiles []string, textColumns []string, textSet map[string]struct{}, outputDir string) {
	var wg sync.WaitGroup
	resultsCh := make(chan []string)

	// Start a worker pool
	for _, csvFile := range csvFiles {
		wg.Add(1)
		go func(csvFile string) {
			defer wg.Done()

			// Print the start message
			fmt.Printf("Searching file: %s\n", csvFile)

			matches, err := searchInCSV(csvFile, textColumns, textSet)
			if err != nil {
				fmt.Printf("Error processing %s: %v\n", csvFile, err)
				return
			}
			if len(matches) > 0 {
				resultsCh <- matches
				outputPath := filepath.Join(outputDir, filepath.Base(csvFile)+"_results.txt")
				if err := saveResults(outputPath, matches); err != nil {
					fmt.Printf("Error saving results for %s: %v\n", csvFile, err)
				}
			}

			// Print the finished message
			fmt.Printf("Finished searching file: %s\n", csvFile)
		}(csvFile)
	}

	// Close the results channel when all workers are done
	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	// Collect results
	for matches := range resultsCh {
		fmt.Printf("Processed file with %d matches\n", len(matches))
	}
}


func main() {
	// Configuration
	jsonlPath := "./shards/shard_00000001_processed.jsonl"
	baseDir := "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/"
	outputDir := "./results"
	textColumns := []string{"en", "tr", "es", "vi"}

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		fmt.Printf("Failed to create output directory: %v\n", err)
		return
	}

	// Build the text set from the JSONL file
	fmt.Println("Building text set...")
	textSet, err := buildTextSet(jsonlPath)
	if err != nil {
		fmt.Printf("Error building text set: %v\n", err)
		return
	}
	fmt.Printf("Text set built with %d entries.\n", len(textSet))

	// Find all CSV files
	fmt.Println("Finding CSV files...")
	csvFiles, err := findCSVFiles(baseDir)
	fmt.Println(csvFiles)
	if err != nil {
		fmt.Printf("Error finding CSV files: %v\n", err)
		return
	}
	fmt.Printf("Found %d CSV files.\n", len(csvFiles))

	// Process the CSV files
	fmt.Println("Processing CSV files...")
	processCSVFiles(csvFiles, textColumns, textSet, outputDir)

	fmt.Println("Processing completed.")
}
