package main

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNormalizeString verifies that normalizeString cleans up text as expected.
func TestNormalizeString(t *testing.T) {
	t.Log("Starting TestNormalizeString")
	input := "  This   is a\n test\tstring  "
	expected := "This is a test string"
	result := normalizeString(input)
	t.Log("Input:", input)
	t.Log("Normalized result:", result)
	if result != expected {
		t.Errorf("Expected %q, got %q", expected, result)
	}
	t.Log("Completed TestNormalizeString")
}

// createTempJSONL creates a temporary JSONL file with the provided records.
func createTempJSONL(t *testing.T, records []JsonlRecord) string {
	t.Helper()
	t.Log("Creating temporary JSONL file")
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "test.jsonl")
	f, err := os.Create(filePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	for _, rec := range records {
		if err := encoder.Encode(rec); err != nil {
			t.Fatal(err)
		}
	}
	t.Log("Temporary JSONL file created at:", filePath)
	return filePath
}

// createTempCSV creates a temporary CSV file with the provided header and rows.
func createTempCSV(t *testing.T, header []string, rows [][]string) string {
	t.Helper()
	t.Log("Creating temporary CSV file")
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "test.csv")
	f, err := os.Create(filePath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	if err := writer.Write(header); err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if err := writer.Write(row); err != nil {
			t.Fatal(err)
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		t.Fatal(err)
	}
	t.Log("Temporary CSV file created at:", filePath)
	return filePath
}

// TestBuildTextSet checks that buildTextSet correctly normalizes and loads JSONL records.
func TestBuildTextSet(t *testing.T) {
	t.Log("Starting TestBuildTextSet")
	records := []JsonlRecord{
		{Text: "  Hello   World!\n"},
		{Text: "Golang\tRocks"},
		{Text: "Already Normalized"},
	}
	jsonlPath := createTempJSONL(t, records)

	textSet, err := buildTextSet(jsonlPath)
	if err != nil {
		t.Fatalf("buildTextSet error: %v", err)
	}
	t.Log("Text set built from JSONL file:", textSet)

	expectedTexts := []string{
		"Hello World!",
		"Golang Rocks",
		"Already Normalized",
	}

	for _, expected := range expectedTexts {
		t.Log("Verifying presence of expected text:", expected)
		if _, ok := textSet[expected]; !ok {
			t.Errorf("Expected text %q not found in textSet", expected)
		}
	}
	t.Log("Completed TestBuildTextSet")
}

// TestSearchCsvForMatches verifies that CSV rows are correctly matched against the JSONL text set.
func TestSearchCsvForMatches(t *testing.T) {
	t.Log("Starting TestSearchCsvForMatches")
	// Create a JSONL file with records that will be normalized.
	records := []JsonlRecord{
		{Text: "Hello World!"},
		{Text: "Goodbye Universe"},
	}
	jsonlPath := createTempJSONL(t, records)
	textSet, err := buildTextSet(jsonlPath)
	if err != nil {
		t.Fatalf("buildTextSet error: %v", err)
	}
	t.Log("Text set from JSONL:", textSet)

	// Create a CSV file. The header includes the target column ("en").
	header := []string{"en", "other"}
	rows := [][]string{
		{"Hello World!", "irrelevant"},
		{"  goodbye universe ", "other info"},
		{"Some random text", "data"},
	}
	csvPath := createTempCSV(t, header, rows)

	t.Log("Searching CSV file for matches in column 'en'")
	matches, err := searchCsvForMatches(csvPath, []string{"en"}, textSet)
	if err != nil {
		t.Fatalf("searchCsvForMatches error: %v", err)
	}
	t.Log("Matches found:", matches)

	// Expected to match both "Hello World!" and the fuzzy match for "goodbye universe".
	expectedMatches := []string{
		"Hello World!",
		"goodbye universe",
	}

	normalizeSlice := func(slice []string) []string {
		var normalized []string
		for _, s := range slice {
			normalized = append(normalized, strings.ToLower(strings.TrimSpace(s)))
		}
		return normalized
	}

	got := normalizeSlice(matches)
	expected := normalizeSlice(expectedMatches)
	t.Log("Normalized expected matches:", expected)
	t.Log("Normalized obtained matches:", got)

	for _, exp := range expected {
		found := false
		for _, g := range got {
			if g == exp {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected match %q not found in results: %v", exp, got)
		}
	}

	if len(got) != len(expected) {
		t.Errorf("Expected %d matches, got %d", len(expected), len(got))
	}
	t.Log("Completed TestSearchCsvForMatches")
}
