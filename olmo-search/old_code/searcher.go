package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Record represents a JSON record in a shard file.
type Record struct {
	Text string `json:"text"`
}

// resultWriter serializes writes to CSV result files.
type resultWriter struct {
	mu sync.Mutex
}

// writeResults appends matching rows to the CSV output file.
// The output file name is constructed from the CSV's base name plus the global shard ID.
func (w *resultWriter) writeResults(csvPath, outputDir, shardOrigin string, matches []string, globalShardID string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	outputFile := fmt.Sprintf("%s_results_shard_%s.csv",
		strings.TrimSuffix(filepath.Base(csvPath), ".csv"),
		globalShardID)
	outputPath := filepath.Join(outputDir, outputFile)

	file, err := os.OpenFile(outputPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header if the file is empty.
	if stat, err := file.Stat(); err == nil && stat.Size() == 0 {
		if err := writer.Write([]string{"text", "shard_origin"}); err != nil {
			return err
		}
	}

	for _, text := range matches {
		if err := writer.Write([]string{text, shardOrigin}); err != nil {
			return err
		}
	}
	return nil
}

// findCSVFiles recursively finds all CSV files under the given root directory.
func findCSVFiles(root string) ([]string, error) {
	var csvFiles []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
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

// isValidShardFile returns true if the filename meets the following criteria:
//   - Begins with "shard_"
//   - Ends with either ".jsonl" or ".jsonl.zst"
//   - The part between the prefix and the suffix is exactly 8 digits.
func isValidShardFile(filename string) bool {
	var suffix string
	if strings.HasSuffix(filename, ".jsonl") {
		suffix = ".jsonl"
	} else if strings.HasSuffix(filename, ".jsonl.zst") {
		suffix = ".jsonl.zst"
	} else {
		return false
	}
	if !strings.HasPrefix(filename, "shard_") {
		return false
	}
	// Extract numeric part.
	numPart := filename[len("shard_") : len(filename)-len(suffix)]
	if len(numPart) != 8 {
		return false
	}
	// Check that every character is a digit.
	for _, r := range numPart {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// findShardFiles recursively crawls the given global shard directory and returns
// all files that are considered valid shard files.
func findShardFiles(globalShardDir string) ([]string, error) {
	var shardFiles []string
	err := filepath.Walk(globalShardDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			base := filepath.Base(path)
			if isValidShardFile(base) {
				shardFiles = append(shardFiles, path)
			}
		}
		return nil
	})
	return shardFiles, err
}

// buildTextSet reads each shard file and builds a set (map with empty struct values)
// containing all texts found.
func buildTextSet(shardFiles []string) (map[string]struct{}, error) {
	textSet := make(map[string]struct{})
	for _, filePath := range shardFiles {
		f, err := os.Open(filePath)
		if err != nil {
			fmt.Printf("Warning: cannot open %s: %v\n", filePath, err)
			continue
		}
		dec := json.NewDecoder(f)
		for dec.More() {
			var rec Record
			if err := dec.Decode(&rec); err != nil {
				continue
			}
			if rec.Text != "" {
				textSet[rec.Text] = struct{}{}
			}
		}
		f.Close()
	}
	return textSet, nil
}

// searchCSV reads the CSV file and returns any values (from the specified columns)
// that exactly match any text in the textSet.
func searchCSV(csvPath string, colList []string, textSet map[string]struct{}) ([]string, error) {
	f, err := os.Open(csvPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	headers, err := reader.Read()
	if err != nil {
		return nil, err
	}

	// Find indices of desired columns (case-insensitive).
	var colIndices []int
	for _, col := range colList {
		for i, header := range headers {
			if strings.EqualFold(header, col) {
				colIndices = append(colIndices, i)
				break
			}
		}
	}

	var matches []string
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		for _, idx := range colIndices {
			if idx < len(record) {
				value := strings.TrimSpace(record[idx])
				if _, exists := textSet[value]; exists {
					matches = append(matches, value)
				}
			}
		}
	}
	return matches, nil
}

// processGlobalShard searches each CSV file for matching texts from the textSet.
// It processes CSV files concurrently (up to 10 at a time).
func processGlobalShard(textSet map[string]struct{}, csvFiles []string, colList []string, outputDir string, writer *resultWriter, globalShardID, globalShardDir string) {
	fmt.Printf("Processing global shard directory: %s (ID: %s)\n", globalShardDir, globalShardID)
	var wg sync.WaitGroup
	sem := make(chan struct{}, 10)

	for _, csvPath := range csvFiles {
		wg.Add(1)
		sem <- struct{}{}
		go func(csvPath string) {
			defer func() {
				<-sem
				wg.Done()
			}()
			// fmt.Printf("Processing CSV: %s\n", csvPath)
			matches, err := searchCSV(csvPath, colList, textSet)
			if err != nil {
				fmt.Printf("Error searching CSV %s: %v\n", csvPath, err)
				return
			}
			if len(matches) > 0 {
				if err := writer.writeResults(csvPath, outputDir, globalShardDir, matches, globalShardID); err != nil {
					fmt.Printf("Error writing results for CSV %s: %v\n", csvPath, err)
				}
			}
		}(csvPath)
	}
	wg.Wait()
}

func main() {
	// Parse the global shard index from the command line.
	indexPtr := flag.Int("index", -1, "Global shard index to process (1-10)")
	flag.Parse()
	if *indexPtr < 1 || *indexPtr > 10 {
		fmt.Println("Please provide a valid global shard index (1-10) using the -index flag")
		os.Exit(1)
	}
	globalShardID := fmt.Sprintf("%02d", *indexPtr)

	// Configure the directories (adjust these as needed).
	baseDir := "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/"
	shardRoot := "/scratch3/workspace/ekoruklu_umass_edu-simple/shard/"
	outputDir := "./results_new"
	colList := []string{"en", "tr", "es", "vi"}

	// Create the output directory if it doesn't exist.
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Find all CSV files in the base directory.
	csvFiles, err := findCSVFiles(baseDir)
	if err != nil {
		fmt.Printf("Error finding CSV files: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Found %d CSV files\n", len(csvFiles))

	// Build the global shard directory path.
	globalShardDir := filepath.Join(shardRoot, fmt.Sprintf("global-shard_%s_of_10", globalShardID))
	info, err := os.Stat(globalShardDir)
	if err != nil || !info.IsDir() {
		fmt.Printf("Global shard directory %s does not exist or is not a directory\n", globalShardDir)
		os.Exit(1)
	}

	// Recursively find all valid shard files in the global shard directory.
	shardFiles, err := findShardFiles(globalShardDir)
	if err != nil {
		fmt.Printf("Error finding shard files: %v\n", err)
		os.Exit(1)
	}
	if len(shardFiles) == 0 {
		fmt.Printf("No valid shard files found in %s\n", globalShardDir)
		os.Exit(1)
	}
	fmt.Printf("Found %d shard files in %s\n", len(shardFiles), globalShardDir)

	// Build the text set from the shard files.
	textSet, err := buildTextSet(shardFiles)
	if err != nil {
		fmt.Printf("Error building text set: %v\n", err)
		os.Exit(1)
	}

	var writer resultWriter
	processGlobalShard(textSet, csvFiles, colList, outputDir, &writer, globalShardID, globalShardDir)
	fmt.Printf("Finished processing global shard (ID: %s): %s\n", globalShardID, globalShardDir)
}
