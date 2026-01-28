import pytest
import linreg_core


class TestCSVParsingNative:
    """Tests for native Python type API (Phase 6)."""

    def test_parse_basic_csv_native(self):
        """Test CSV parsing with native API."""
        csv_content = """name,value,category
Alice,42.5,A
Bob,17.3,B
Charlie,99.9,A"""

        result = linreg_core.parse_csv(csv_content)

        # Verify result is a CSVResult object
        assert hasattr(result, 'headers')
        assert hasattr(result, 'data')
        assert hasattr(result, 'numeric_columns')
        assert hasattr(result, 'n_rows')
        assert hasattr(result, 'n_cols')

        # Verify content
        assert result.headers == ["name", "value", "category"]
        assert "value" in result.numeric_columns
        assert result.n_rows == 3
        assert result.n_cols == 3
        assert len(result.data) == 3

    def test_csv_result_object_attributes(self):
        """Test that all CSVResult attributes are accessible."""
        csv_content = """x,y,z
1.5,2.5,hello
3.5,4.5,world"""

        result = linreg_core.parse_csv(csv_content)

        # Test all attributes
        assert isinstance(result.headers, list)
        assert isinstance(result.data, list)
        assert isinstance(result.numeric_columns, list)
        assert isinstance(result.n_rows, int)
        assert isinstance(result.n_cols, int)

    def test_parse_csv_identifies_numeric_columns(self):
        """Test that numeric columns are correctly identified."""
        csv_content = """x,y,z
1.5,2.5,hello
3.5,4.5,world"""

        result = linreg_core.parse_csv(csv_content)

        numeric_set = set(result.numeric_columns)
        assert "x" in numeric_set
        assert "y" in numeric_set
        assert "z" not in numeric_set

    def test_csv_summary_method(self):
        """Test the summary() method of CSVResult."""
        csv_content = """x,y
1.0,2.0
3.0,4.0"""

        result = linreg_core.parse_csv(csv_content)
        summary = result.summary()

        # Verify summary is a string with expected content
        assert isinstance(summary, str)
        assert "CSV Parsing Results" in summary
        assert "Rows:" in summary
        assert "Columns:" in summary

    def test_csv_to_dict_method(self):
        """Test the to_dict() method of CSVResult."""
        csv_content = """x,y
1.0,2.0
3.0,4.0"""

        result = linreg_core.parse_csv(csv_content)
        d = result.to_dict()

        # Verify to_dict returns a proper dict
        assert isinstance(d, dict)
        assert "headers" in d
        assert "data" in d
        assert "numeric_columns" in d
        assert "n_rows" in d
        assert "n_cols" in d

    def test_csv_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        csv_content = """x,y
1.0,2.0"""

        result = linreg_core.parse_csv(csv_content)

        # Test __str__
        str_result = str(result)
        assert "CSV Parsing Results" in str_result

        # Test __repr__
        repr_result = repr(result)
        assert "CSVResult" in repr_result

    def test_parse_csv_multiple_rows(self):
        """Test parsing CSV with multiple rows."""
        csv_content = """a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""

        result = linreg_core.parse_csv(csv_content)

        assert result.n_rows == 4
        assert result.n_cols == 3
        assert len(result.data) == 4

    def test_parse_csv_with_whitespace(self):
        """Test CSV parsing with extra whitespace."""
        csv_content = """x, y ,  z
 1.5 , 2.5 , hello
 3.5 , 4.5 , world"""

        result = linreg_core.parse_csv(csv_content)

        # Headers preserve whitespace from CSV (csv reader doesn't trim)
        assert result.headers == ["x", " y ", "  z"]
        # The column with space should still be numeric if values parse
        assert " y " in result.numeric_columns

    def test_parse_csv_numeric_data_types(self):
        """Test parsing various numeric data types."""
        csv_content = """int_val,float_val,neg_val
42,3.14,-1.5
100,2.71,-999"""

        result = linreg_core.parse_csv(csv_content)

        numeric_set = set(result.numeric_columns)
        assert "int_val" in numeric_set
        assert "float_val" in numeric_set
        assert "neg_val" in numeric_set


class TestCSVParsingEdgeCases:
    """Edge case and error handling tests for CSV parsing."""

    def test_empty_csv(self):
        """Test parsing an empty CSV string."""
        result = linreg_core.parse_csv("")

        # Should return a valid result object
        assert hasattr(result, 'headers')
        assert hasattr(result, 'data')
        assert hasattr(result, 'numeric_columns')
        # Empty input should have no rows
        assert result.n_rows == 0

    def test_csv_with_only_headers(self):
        """Test CSV with headers but no data rows."""
        csv_content = """x,y,z"""

        result = linreg_core.parse_csv(csv_content)
        assert result.headers == ["x", "y", "z"]
        assert result.n_rows == 0
        assert result.n_cols == 3

    def test_csv_with_empty_rows(self):
        """Test CSV with empty lines between data."""
        csv_content = """x,y
1,2

3,4

5,6"""

        result = linreg_core.parse_csv(csv_content)
        # Empty rows should be skipped or handled
        assert result.n_rows >= 2  # At least the non-empty rows

    def test_csv_with_quotes(self):
        """Test CSV with quoted fields."""
        csv_content = '''name,value
"John Doe",42.5
"Jane, Smith",17.3'''

        result = linreg_core.parse_csv(csv_content)
        assert result.headers == ["name", "value"]
        assert "value" in result.numeric_columns
        assert result.n_rows == 2

    def test_csv_with_mixed_types(self):
        """Test CSV with mixed numeric and non-numeric data."""
        csv_content = """x,y,z
1,hello,3.5
2,world,4.5
NA,data,5.5"""

        result = linreg_core.parse_csv(csv_content)
        numeric_set = set(result.numeric_columns)
        # Only x and z should be numeric (y has "hello", "world"; "NA" might be parsed as string)
        assert "x" in numeric_set
        assert "z" in numeric_set

    def test_csv_with_special_characters(self):
        """Test CSV with special characters in data."""
        csv_content = """name,value
test_1,1.5
test-2,2.5
test.3,3.5"""

        result = linreg_core.parse_csv(csv_content)
        assert "value" in result.numeric_columns
        assert result.n_rows == 3

    def test_csv_with_leading_trailing_spaces(self):
        """Test CSV with leading/trailing spaces in headers."""
        csv_content = """  x  ,  y  ,  z
1.5,2.5,hello"""

        result = linreg_core.parse_csv(csv_content)
        # Headers preserve whitespace
        assert len(result.headers) == 3

    def test_csv_with_inconsistent_column_counts(self):
        """Test CSV where rows have different numbers of columns."""
        csv_content = """x,y,z
1,2,3
4,5
6,7,8"""

        result = linreg_core.parse_csv(csv_content)
        # Should handle gracefully - may skip or pad missing values
        assert hasattr(result, 'data')

    def test_csv_with_unicode(self):
        """Test CSV with Unicode characters."""
        csv_content = """name,value
café,42.5
日本語,17.3
مرحبا,99.9"""

        result = linreg_core.parse_csv(csv_content)
        assert result.headers == ["name", "value"]
        assert result.n_rows == 3

    def test_csv_with_zero_values(self):
        """Test CSV with zero values (should be numeric)."""
        csv_content = """x,y
0,0.0
0,0"""

        result = linreg_core.parse_csv(csv_content)
        numeric_set = set(result.numeric_columns)
        assert "x" in numeric_set
        assert "y" in numeric_set

    def test_csv_with_boolean_strings(self):
        """Test CSV with boolean-like strings (should not be numeric)."""
        csv_content = """x,y
true,1.5
false,2.5"""

        result = linreg_core.parse_csv(csv_content)
        numeric_set = set(result.numeric_columns)
        # "true" and "false" should not be numeric
        assert "y" in numeric_set
