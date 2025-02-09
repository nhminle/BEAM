package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExactMatch(t *testing.T) {
	// Create temporary directories for CSV files, shard files, and output.
	baseDir := t.TempDir()   // will act as our "Prompts" directory for CSV files
	shardDir := t.TempDir()  // will act as our "shard" directory
	outputDir := t.TempDir() // will hold our results

	// -------------------------------
	// Create a test CSV file.
	// -------------------------------
	// The CSV has headers: en,tr,es,vi and one row with "Hello World" in the en column.
	csvContent := `en,tr,es,vi
Hello World,Merhaba,Hola,Xin chào`
	csvFilePath := filepath.Join(baseDir, "test.csv")
	if err := os.WriteFile(csvFilePath, []byte(csvContent), 0644); err != nil {
		t.Fatalf("Error writing CSV file: %v", err)
	}

	// -------------------------------
	// Create a test shard file.
	// -------------------------------
	// The shard file is named "01.jsonl" and includes a record with text "Hello World"
	// (which should exactly match the CSV value) plus another non-matching record.
	shardContent := `{"text": "Hello World"}
{"text": "Not Matching"}`
	shardFilePath := filepath.Join(shardDir, "01.jsonl")
	if err := os.WriteFile(shardFilePath, []byte(shardContent), 0644); err != nil {
		t.Fatalf("Error writing shard file: %v", err)
	}

	// -------------------------------
	// Find CSV files.
	// -------------------------------
	// Instead of using findCSVFiles over a large tree, we call it on our temporary baseDir.
	csvFiles, err := findCSVFiles(baseDir)
	if err != nil {
		t.Fatalf("Error finding CSV files: %v", err)
	}
	if len(csvFiles) == 0 {
		t.Fatal("No CSV files found")
	}

	// -------------------------------
	// Run the processing for the test shard.
	// -------------------------------
	// Our test shard has a filename that starts with "01". We use the same shard id.
	shardID := "01"
	var writer resultWriter
	textColumns := []string{"en", "tr", "es", "vi"}

	// Instead of searching for the shard in a directory,
	// we directly supply our temporary shard file.
	processShard(shardFilePath, csvFiles, textColumns, outputDir, &writer, shardID)

	// -------------------------------
	// Check the output.
	// -------------------------------
	// The expected result file should be named "test_results_shard_01.csv"
	resultFilePath := filepath.Join(outputDir, "test_results_shard_01.csv")
	data, err := os.ReadFile(resultFilePath)
	if err != nil {
		t.Fatalf("Error reading result file: %v", err)
	}
	content := string(data)

	// We expect the file to contain the header plus a row containing "Hello World"
	if !strings.Contains(content, "Hello World") {
		t.Errorf("Expected 'Hello World' in result file, got:\n%s", content)
	}
}
