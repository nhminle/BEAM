package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/text/unicode/norm"
)

// normalizeString cleans up a string by normalizing unicode and replacing
// all whitespace sequences (newlines, tabs, multiple spaces) with a single space.
func normalizeString(s string) string {
	// Normalize the string using NFKC normalization.
	s = norm.NFKC.String(s)
	// Replace all whitespace (newlines, tabs, multiple spaces) with a single space.
	re := regexp.MustCompile(`\s+`)
	s = re.ReplaceAllString(s, " ")
	// Trim any leading or trailing spaces.
	return strings.TrimSpace(s)
}

// JsonlRecord represents one JSONL record from a shard.
type JsonlRecord struct {
	Text string `json:"text"`
}

// ResultFileWriter serializes writes to result files.
type ResultFileWriter struct {
	mu sync.Mutex
}

// appendResultsForCsv opens (or creates) a text file for the given CSV file (book)
// and appends the matching results. If the file is new (or empty), it writes a header line.
func (w *ResultFileWriter) appendResultsForCsv(csvFilePath, outputDirectory string, matches []CsvMatch) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	outputFilePath := filepath.Join(outputDirectory,
		fmt.Sprintf("%s_results.txt",
			strings.TrimSuffix(filepath.Base(csvFilePath), ".csv"),
		))
	// Check if the file exists.
	_, err := os.Stat(outputFilePath)
	fileExists := err == nil

	// Open the file in append mode (or create it).
	file, err := os.OpenFile(outputFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	// If the file is new, write the header.
	if !fileExists {
		if _, err := file.WriteString("text,shard_file_path\n"); err != nil {
			return err
		}
	}

	// Append each match.
	for _, match := range matches {
		line := fmt.Sprintf("%s,%s\n", match.Text, match.ShardFilePath)
		if _, err := file.WriteString(line); err != nil {
			return err
		}
	}
	return nil
}

// CsvMatch represents one matching row from a shard.
type CsvMatch struct {
	Text          string
	ShardFilePath string
}

// levenshtein computes the Levenshtein distance between two strings.
func levenshtein(a, b string) int {
	lenA, lenB := len(a), len(b)
	// Create a 2D slice.
	dp := make([][]int, lenA+1)
	for i := 0; i <= lenA; i++ {
		dp[i] = make([]int, lenB+1)
		dp[i][0] = i
	}
	for j := 0; j <= lenB; j++ {
		dp[0][j] = j
	}

	for i := 1; i <= lenA; i++ {
		for j := 1; j <= lenB; j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			dp[i][j] = int(math.Min(
				math.Min(float64(dp[i-1][j])+1, float64(dp[i][j-1])+1),
				float64(dp[i-1][j-1]+cost),
			))
		}
	}
	return dp[lenA][lenB]
}

// fuzzyMatch returns true if the similarity ratio between s1 and s2 is at least 80%.
func fuzzyMatch(s1, s2 string) bool {
	distance := levenshtein(s1, s2)
	maxLength := math.Max(float64(len(s1)), float64(len(s2)))
	if maxLength == 0 {
		return true
	}
	ratio := 1 - float64(distance)/maxLength
	return ratio >= 0.8
}

// findShardFiles finds all files with a .jsonl extension under the given root directory.
func findShardFiles(rootDirectory string) ([]string, error) {
	var shardFilePaths []string
	err := filepath.Walk(rootDirectory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".jsonl") {
			shardFilePaths = append(shardFilePaths, path)
		}
		return nil
	})
	return shardFilePaths, err
}

// findCSVFiles finds all files with a .csv extension under the given root directory.
func findCSVFiles(rootDirectory string) ([]string, error) {
	var csvFilePaths []string
	err := filepath.Walk(rootDirectory, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".csv") {
			csvFilePaths = append(csvFilePaths, path)
		}
		return nil
	})
	return csvFilePaths, err
}

// buildTextSet constructs a set of normalized texts from a shard (JSONL) file.
func buildTextSet(shardFilePath string) (map[string]struct{}, error) {
	file, err := os.Open(shardFilePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	textSet := make(map[string]struct{})
	decoder := json.NewDecoder(file)
	for decoder.More() {
		var record JsonlRecord
		if err := decoder.Decode(&record); err != nil {
			// Skip decode errors.
			continue
		}
		if record.Text != "" {
			// Normalize the text to clean up any inline or unusual characters.
			cleanText := normalizeString(record.Text)
			textSet[cleanText] = struct{}{}
		}
	}
	return textSet, nil
}

// searchCsvForMatches reads csvFilePath, searches the specified columns for texts that fuzzy match
// any key in textSet (with at least 80% similarity), and returns any matching texts.
func searchCsvForMatches(csvFilePath string, targetColumns []string, textSet map[string]struct{}) ([]string, error) {
	file, err := os.Open(csvFilePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	headers, err := reader.Read()
	if err != nil {
		return nil, err
	}

	// Identify which column indices to search (case-insensitive).
	var targetColumnIndices []int
	for _, targetCol := range targetColumns {
		for i, header := range headers {
			if strings.EqualFold(header, targetCol) {
				targetColumnIndices = append(targetColumnIndices, i)
				break
			}
		}
	}

	var matchingTexts []string
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		for _, idx := range targetColumnIndices {
			if idx < len(record) {
				cellText := strings.TrimSpace(record[idx])
				// Fuzzy match: iterate over keys in textSet.
				for key := range textSet {
					if fuzzyMatch(key, cellText) {
						matchingTexts = append(matchingTexts, cellText)
						break // Found a match; move to the next column.
					}
				}
			}
		}
	}
	return matchingTexts, nil
}

// processShard processes one shard (JSONL file):
// 1. It builds the set of normalized texts from the shard.
// 2. For each CSV file, it searches for matching texts.
// 3. It immediately appends the results to the corresponding CSV result file.
func processShard(shardFilePath string, csvFilePaths []string, targetColumns []string, outputDirectory string, writer *ResultFileWriter) {
	fmt.Printf("Processing shard: %s\n", shardFilePath)
	shardTextSet, err := buildTextSet(shardFilePath)
	if err != nil {
		fmt.Printf("Error building text set for %s: %v\n", shardFilePath, err)
		return
	}

	var wg sync.WaitGroup
	// Limit concurrent CSV processing within this shard.
	csvWorkerSemaphore := make(chan struct{}, 10)

	for _, csvFilePath := range csvFilePaths {
		wg.Add(1)
		csvWorkerSemaphore <- struct{}{}
		go func(csvPath string) {
			defer func() {
				<-csvWorkerSemaphore
				wg.Done()
			}()

			fmt.Printf("Processing CSV file: %s (in shard: %s)\n", csvPath, shardFilePath)
			matchingTexts, err := searchCsvForMatches(csvPath, targetColumns, shardTextSet)
			if err != nil {
				fmt.Printf("Error searching %s: %v\n", csvPath, err)
				return
			}

			// Convert matches to CsvMatch entries.
			var matches []CsvMatch
			for _, text := range matchingTexts {
				matches = append(matches, CsvMatch{Text: text, ShardFilePath: shardFilePath})
			}

			// Immediately append the results for this CSV file.
			if len(matches) > 0 {
				if err := writer.appendResultsForCsv(csvPath, outputDirectory, matches); err != nil {
					fmt.Printf("Error appending results for %s: %v\n", csvPath, err)
				} else {
					fmt.Printf("Appended results for %s from shard %s\n", csvPath, shardFilePath)
				}
			}
		}(csvFilePath)
	}
	wg.Wait()
}

func main() {
	// Adjust these paths as needed.
	baseDirectory := "/home/ekorukluoglu_umass_edu/beam2/BEAM/olmo-search/sample-set/"
	shardRootDirectory := "/scratch3/workspace/ekorukluoglu_umass_edu-simple/dclm/global-shard_01_of_10/local-shard_0_of_10/"
	outputDirectory := "./results"
	targetTextColumns := []string{"en", "tr", "es", "vi"}

	// Create the output directory.
	if err := os.MkdirAll(outputDirectory, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		return
	}

	// Find all CSV files (books) recursively.
	csvFilePaths, err := findCSVFiles(baseDirectory)
	if err != nil {
		fmt.Printf("Error finding CSV files: %v\n", err)
		return
	}
	fmt.Printf("Found %d CSV files\n", len(csvFilePaths))

	// Find all shard files.
	shardFilePaths, err := findShardFiles(shardRootDirectory)
	if err != nil {
		fmt.Printf("Error finding shard files: %v\n", err)
		return
	}
	fmt.Printf("Found %d shard files\n", len(shardFilePaths))

	// Prepare a result writer.
	var writer ResultFileWriter

	// Process shards concurrently.
	var shardWg sync.WaitGroup
	shardChannel := make(chan string, len(shardFilePaths))

	// Start a fixed number of shard workers (e.g. 40).
	numShardWorkers := 40
	for i := 0; i < numShardWorkers; i++ {
		go func() {
			for shardPath := range shardChannel {
				processShard(shardPath, csvFilePaths, targetTextColumns, outputDirectory, &writer)
				shardWg.Done()
			}
		}()
	}

	// Enqueue all shard files.
	for _, shardPath := range shardFilePaths {
		shardWg.Add(1)
		shardChannel <- shardPath
	}
	close(shardChannel)

	// Wait for all shards to finish processing.
	shardWg.Wait()
	fmt.Printf("Processed all shards. Results have been appended per CSV file in: %s\n", outputDirectory)
}
