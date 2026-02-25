Attribute VB_Name = "ExampleMacros"
' ==============================================================================
' ExampleMacros.bas - Setup and button macros for LinregCore_Example.xlsm
' ==============================================================================
' END USER FLOW (after importing both .bas files and adding Workbook_Open):
'   1. Open the workbook  ->  LR_Init fires automatically
'   2. Run SetupWorkbook() once from the VBA editor (Alt+F11 > Run > Run Macro)
'   3. All sheets, data, labels, and buttons are created automatically
'   4. Click any button - done
' ==============================================================================
Option Explicit

' -- Sheet names ---------------------------------------------------------------
Private Const SH_OLS   As String = "OLS Example"
Private Const SH_DIAG  As String = "Diagnostics"
Private Const SH_REG   As String = "Regularized"
Private Const SH_CV    As String = "Cross Validation"
Private Const SH_POLY  As String = "Polynomial"
Private Const SH_INSTR As String = "Instructions"

' -- Data location (OLS Example sheet) ----------------------------------------
Private Const Y_COL   As Long = 1   ' col A  (mpg)
Private Const X_FIRST As Long = 2   ' col B  (first predictor)
Private Const X_LAST  As Long = 5   ' col E  (last predictor)
Private Const HDR_ROW As Long = 3   ' header row
Private Const DATA_R1 As Long = 4   ' first data row
Private Const DATA_R2 As Long = 23  ' last  data row  (20 observations)

' -- Output anchor (OLS results go to col G) -----------------------------------
Private Const OLS_OUT_COL As Long = 7
Private Const OLS_OUT_ROW As Long = 1

' ==============================================================================
' SECTION 1 - Sheet / range helpers
' ==============================================================================

Private Function GetOrCreateSheet(wsName As String) As Worksheet
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Sheets(wsName)
    On Error GoTo 0
    If ws Is Nothing Then
        Set ws = ThisWorkbook.Sheets.Add( _
            After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
        ws.Name = wsName
    End If
    Set GetOrCreateSheet = ws
End Function

Private Function YRange() As Range
    With ThisWorkbook.Sheets(SH_OLS)
        Set YRange = .Range(.Cells(DATA_R1, Y_COL), .Cells(DATA_R2, Y_COL))
    End With
End Function

Private Function XRange() As Range
    With ThisWorkbook.Sheets(SH_OLS)
        Set XRange = .Range(.Cells(DATA_R1, X_FIRST), .Cells(DATA_R2, X_LAST))
    End With
End Function

' ==============================================================================
' SECTION 2 - Formatting helpers
' ==============================================================================

Private Sub StyleHeader(ws As Worksheet, startRow As Long, _
                         startCol As Long, nCols As Long)
    With ws.Range(ws.Cells(startRow, startCol), _
                  ws.Cells(startRow, startCol + nCols - 1))
        .Font.Bold = True
        .Interior.Color = RGB(68, 114, 196)
        .Font.Color = RGB(255, 255, 255)
    End With
End Sub

Private Sub AddButton(ws As Worksheet, caption As String, macroName As String, _
                       anchorCell As Range, Optional btnWidth As Double = 130, _
                       Optional btnHeight As Double = 28)
    Dim btn As Button
    Set btn = ws.Buttons.Add(anchorCell.Left, anchorCell.Top, btnWidth, btnHeight)
    With btn
        .Caption = macroName  ' temp - overwritten below
        .Caption = caption
        .OnAction = macroName
        .Font.Size = 10
        .Font.Name = "Calibri"
    End With
End Sub

' ==============================================================================
' SECTION 3 - Paste result helper
' ==============================================================================

' Paste a 2D result array into ws at (startRow, startCol).
' Clears a 100×10 block first.  Writes an error message for 1D error returns.
Private Sub PasteResult(ws As Worksheet, result As Variant, _
                         startRow As Long, startCol As Long)
    ws.Range(ws.Cells(startRow, startCol), _
             ws.Cells(startRow + 100, startCol + 9)).ClearContents

    If Not IsArray(result) Then
        ws.Cells(startRow, startCol).Value = CStr(result)
        Exit Sub
    End If

    Dim is2D As Boolean
    is2D = False
    On Error Resume Next
    Dim chk As Long: chk = UBound(result, 2)
    If Err.Number = 0 Then is2D = True
    Err.Clear
    On Error GoTo 0

    If Not is2D Then
        ws.Cells(startRow, startCol).Value = "ERROR: " & CStr(result(0))
        Exit Sub
    End If

    Dim nR As Long, nC As Long
    nR = UBound(result, 1) - LBound(result, 1) + 1
    nC = UBound(result, 2) - LBound(result, 2) + 1
    ws.Range(ws.Cells(startRow, startCol), _
             ws.Cells(startRow + nR - 1, startCol + nC - 1)).Value = result
End Sub

' ==============================================================================
' SECTION 4 - SetupWorkbook  (run once after importing the two .bas files)
' ==============================================================================

Public Sub SetupWorkbook()
    Application.ScreenUpdating = False

    SetupOLSSheet
    SetupDiagSheet
    SetupRegSheet
    SetupCVSheet
    SetupPolySheet
    SetupInstrSheet

    ' Land on the OLS sheet when done
    ThisWorkbook.Sheets(SH_OLS).Activate

    Application.ScreenUpdating = True
    MsgBox "Workbook setup complete!" & vbCrLf & vbCrLf & _
           "Click any button on the sheets to run the models." & vbCrLf & _
           "(Make sure LR_Init has been called - it fires from Workbook_Open.)", _
           vbInformation, "LinregCore Setup"
End Sub

' -- OLS Example sheet ---------------------------------------------------------
Private Sub SetupOLSSheet()
    Dim ws As Worksheet
    Set ws = GetOrCreateSheet(SH_OLS)
    ws.Cells.ClearContents
    ws.Cells.ClearFormats
    ws.Buttons.Delete

    ' Title
    With ws.Cells(1, 1)
        .Value = "LinregCore - OLS Regression Example"
        .Font.Bold = True
        .Font.Size = 13
    End With
    ws.Cells(2, 1).Value = "Dataset: mtcars (first 20 rows)   |   Y = mpg   |   X = cyl, disp, hp, wt"
    ws.Cells(2, 1).Font.Italic = True

    ' Headers row 3
    ws.Cells(HDR_ROW, 1).Value = "mpg"
    ws.Cells(HDR_ROW, 2).Value = "cyl"
    ws.Cells(HDR_ROW, 3).Value = "disp"
    ws.Cells(HDR_ROW, 4).Value = "hp"
    ws.Cells(HDR_ROW, 5).Value = "wt"
    StyleHeader ws, HDR_ROW, 1, 5

    ' mtcars data (first 20 rows)
    Dim mpgD  As Variant
    Dim cylD  As Variant
    Dim dispD As Variant
    Dim hpD   As Variant
    Dim wtD   As Variant

    mpgD  = Array(21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, _
                  22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4, _
                  14.7, 32.4, 30.4, 33.9)
    cylD  = Array(6, 6, 4, 6, 8, 6, 8, 4, 4, 6, 6, 8, 8, 8, 8, 8, 8, 4, 4, 4)
    dispD = Array(160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, _
                  140.8, 167.6, 167.6, 275.8, 275.8, 275.8, 472.0, 460.0, _
                  440.0, 78.7, 75.7, 71.1)
    hpD   = Array(110, 110, 93, 110, 175, 105, 245, 62, 95, 123, _
                  123, 180, 180, 180, 205, 215, 230, 66, 52, 65)
    wtD   = Array(2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, _
                  3.150, 3.440, 3.440, 4.070, 3.730, 3.780, 5.250, 5.424, _
                  5.345, 2.200, 1.615, 1.835)

    Dim i As Long
    For i = 0 To 19
        ws.Cells(DATA_R1 + i, 1).Value = mpgD(i)
        ws.Cells(DATA_R1 + i, 2).Value = cylD(i)
        ws.Cells(DATA_R1 + i, 3).Value = dispD(i)
        ws.Cells(DATA_R1 + i, 4).Value = hpD(i)
        ws.Cells(DATA_R1 + i, 5).Value = wtD(i)
    Next i

    ' Alternating row shading on data rows
    For i = 0 To 19
        If i Mod 2 = 0 Then
            ws.Range(ws.Cells(DATA_R1 + i, 1), ws.Cells(DATA_R1 + i, 5)) _
              .Interior.Color = RGB(235, 241, 252)
        End If
    Next i

    ' Column widths for data area
    ws.Columns(1).ColumnWidth = 8
    ws.Columns(2).ColumnWidth = 7
    ws.Columns(3).ColumnWidth = 8
    ws.Columns(4).ColumnWidth = 7
    ws.Columns(5).ColumnWidth = 8
    ws.Columns(6).ColumnWidth = 12  ' Weights column

    ' Buttons in col G (spacer before buttons)
    AddButton ws, "Run OLS",              "RunOLS",               ws.Cells(12, 7), 140, 28
    AddButton ws, "Prediction Intervals", "RunPredictionIntervals", ws.Cells(14, 7), 140, 28
    AddButton ws, "VIF Test",             "RunVIF",               ws.Cells(16, 7), 140, 28
    AddButton ws, "WLS (Weighted)",       "RunWLS",               ws.Cells(18, 7), 140, 28
End Sub

' -- Diagnostics sheet ---------------------------------------------------------
Private Sub SetupDiagSheet()
    Dim ws As Worksheet
    Set ws = GetOrCreateSheet(SH_DIAG)
    ws.Cells.ClearContents
    ws.Cells.ClearFormats
    ws.Buttons.Delete

    With ws.Cells(1, 1)
        .Value = "LinregCore - Diagnostic Tests"
        .Font.Bold = True
        .Font.Size = 13
    End With
    ws.Cells(2, 1).Value = _
        "Runs 10 tests on the mtcars data from the OLS Example sheet."
    ws.Cells(2, 1).Font.Italic = True

    ' Button at row 1 col C so it doesn't overlap the output table header
    AddButton ws, "Run All Diagnostics", "RunAllDiagnostics", ws.Cells(3, 6), 160, 28

    ' Column widths (results table written by RunAllDiagnostics)
    ws.Columns(1).ColumnWidth = 26
    ws.Columns(2).ColumnWidth = 14
    ws.Columns(3).ColumnWidth = 16
    ws.Columns(4).ColumnWidth = 8
End Sub

' -- Regularized sheet ---------------------------------------------------------
Private Sub SetupRegSheet()
    Dim ws As Worksheet
    Set ws = GetOrCreateSheet(SH_REG)
    ws.Cells.ClearContents
    ws.Cells.ClearFormats
    ws.Buttons.Delete

    With ws.Cells(1, 1)
        .Value = "LinregCore - Regularized Regression"
        .Font.Bold = True
        .Font.Size = 13
    End With

    ' Lambda / Alpha input cells
    ws.Cells(2, 1).Value = "Lambda:"
    ws.Cells(2, 1).Font.Bold = True
    ws.Cells(2, 2).Value = 1#             ' default lambda = 1.0
    ws.Cells(2, 2).Interior.Color = RGB(255, 255, 200)

    ws.Cells(3, 1).Value = "Alpha (0 = Ridge, 1 = Lasso):"
    ws.Cells(3, 1).Font.Bold = True
    ws.Cells(3, 2).Value = 0.5            ' default alpha = 0.5
    ws.Cells(3, 2).Interior.Color = RGB(255, 255, 200)

    ws.Cells(4, 1).Value = _
        "Edit the yellow cells, then click a button."
    ws.Cells(4, 1).Font.Italic = True
    ws.Cells(4, 1).Font.Color = RGB(100, 100, 100)

    ' Buttons (col A rows 6-8, output to col D onwards)
    AddButton ws, "Ridge",       "RunRidge",       ws.Cells(6, 1), 120, 28
    AddButton ws, "Lasso",       "RunLasso",       ws.Cells(9, 1), 120, 28
    AddButton ws, "Elastic Net", "RunElasticNet",  ws.Cells(12, 1), 120, 28

    ' Column widths
    ws.Columns(1).ColumnWidth = 30
    ws.Columns(2).ColumnWidth = 10
    ws.Columns(3).ColumnWidth = 14
    ws.Columns(4).ColumnWidth = 14
    ws.Columns(5).ColumnWidth = 14
End Sub

' -- Cross Validation sheet -------------------------------------------------
Private Sub SetupCVSheet()
    Dim ws As Worksheet
    Set ws = GetOrCreateSheet(SH_CV)
    ws.Cells.ClearContents
    ws.Cells.ClearFormats
    ws.Buttons.Delete

    With ws.Cells(1, 1)
        .Value = "LinregCore - K-Fold Cross Validation"
        .Font.Bold = True
        .Font.Size = 13
    End With
    ws.Cells(2, 1).Value = _
        "K-Fold CV evaluates model performance by splitting data into folds, training on subsets, and testing on held-out folds."

    ' CV options
    ws.Cells(4, 1).Value = "Number of Folds:"
    ws.Cells(4, 1).Font.Bold = True
    ws.Cells(4, 2).Value = 5
    ws.Cells(4, 2).Interior.Color = RGB(255, 255, 200)

    ' Buttons in column H (after output data)
    AddButton ws, "KFold OLS", "RunKFoldOLS", ws.Cells(6, 8), 140, 28
    AddButton ws, "KFold Ridge", "RunKFoldRidge", ws.Cells(9, 8), 140, 28
    AddButton ws, "KFold Lasso", "RunKFoldLasso", ws.Cells(12, 8), 140, 28
    AddButton ws, "KFold Elastic Net", "RunKFoldElasticNet", ws.Cells(15, 8), 140, 28

    ' Column widths
    ws.Columns(1).ColumnWidth = 30
    ws.Columns(2).ColumnWidth = 12
    ws.Columns(3).ColumnWidth = 14
End Sub

' -- Polynomial sheet ---------------------------------------------------------
Private Sub SetupPolySheet()
    Dim ws As Worksheet
    Set ws = GetOrCreateSheet(SH_POLY)
    ws.Cells.ClearContents
    ws.Cells.ClearFormats
    ws.Buttons.Delete

    With ws.Cells(1, 1)
        .Value = "LinregCore - Polynomial Regression"
        .Font.Bold = True
        .Font.Size = 13
    End With
    ws.Cells(2, 1).Value = _
        "Fit y = b0 + b1*x + b2*x^2 + ... + bd*x^d. Uses single predictor x from OLS sheet (hp)."

    ' Create sample data: quadratic relationship y = 1 + 2*x + 0.5*x^2
    Dim i As Long
    Dim xVal As Double
    ws.Cells(4, 1).Value = "x"
    ws.Cells(4, 2).Value = "y"
    StyleHeader ws, 4, 1, 2

    For i = 1 To 20
        xVal = i * 5  ' 5, 10, 15, ..., 100
        ws.Cells(4 + i, 1).Value = xVal
        ws.Cells(4 + i, 2).Value = 1 + 2 * xVal + 0.5 * xVal * xVal + (Rnd() - 0.5) * 50
    Next i

    ' Alternating row shading
    For i = 1 To 20
        If i Mod 2 = 0 Then
            ws.Range(ws.Cells(4 + i, 1), ws.Cells(4 + i, 2)).Interior.Color = RGB(235, 241, 252)
        End If
    Next i

    ws.Columns(1).ColumnWidth = 10
    ws.Columns(2).ColumnWidth = 12

    ' Options
    ws.Cells(27, 1).Value = "Degree:"
    ws.Cells(27, 1).Font.Bold = True
    ws.Cells(27, 2).Value = 2
    ws.Cells(27, 2).Interior.Color = RGB(255, 255, 200)

    ws.Cells(28, 1).Value = "Center (recommended for degree >= 3):"
    ws.Cells(28, 1).Font.Bold = True
    ws.Cells(28, 2).Value = False
    ws.Cells(28, 2).Interior.Color = RGB(255, 255, 200)

    ' Wrap explanation to next line for readability
    ws.Cells(29, 1).Value = "Reduces multicollinearity between x, x^2, x^3, etc."

    ' Buttons
    AddButton ws, "Fit Polynomial", "RunPolynomial", ws.Cells(32, 1), 140, 28
    AddButton ws, "Fit Quadratic (deg=2)", "RunPolynomialQuad", ws.Cells(35, 1), 140, 28
    AddButton ws, "Fit Cubic (deg=3)", "RunPolynomialCubic", ws.Cells(38, 1), 140, 28

    ' Column widths for output area
    ws.Columns(4).ColumnWidth = 11  ' Coefficient column
    ws.Columns(1).ColumnWidth = 10   ' Ensure term column is wide enough
End Sub

' -- Instructions sheet --------------------------------------------------------
Private Sub SetupInstrSheet()
    Dim ws As Worksheet
    Set ws = GetOrCreateSheet(SH_INSTR)
    ws.Cells.ClearContents
    ws.Cells.ClearFormats

    Dim r As Long
    r = 1

    ws.Cells(r, 1).Value = "LinregCore - Setup & Usage Guide"
    ws.Cells(r, 1).Font.Bold = True
    ws.Cells(r, 1).Font.Size = 14
    r = r + 2

    ws.Cells(r, 1).Value = "WHAT IS THIS WORKBOOK?"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = _
        "This workbook connects Excel to the linreg_core.dll statistical library, which " & _
        "performs OLS, Ridge, Lasso, and Elastic Net regression plus 10 diagnostic tests."
    r = r + 2

    ws.Cells(r, 1).Value = "FILES REQUIRED (place all three in the same folder as this workbook)"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   linreg_core_x64.dll  - for 64-bit Excel (most modern installs)"
    r = r + 1
    ws.Cells(r, 1).Value = "   linreg_core_x86.dll  - for 32-bit Excel"
    r = r + 1
    ws.Cells(r, 1).Value = "   LinregCore_Example.xlsm  - this workbook"
    r = r + 2

    ws.Cells(r, 1).Value = "FIRST-TIME SETUP (one-time only)"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   1. Place both DLL files and this workbook in the same folder."
    r = r + 1
    ws.Cells(r, 1).Value = "   2. Open the workbook - click Enable Macros if prompted."
    r = r + 1
    ws.Cells(r, 1).Value = "   3. That's it. The library loads automatically on open."
    r = r + 2

    ws.Cells(r, 1).Value = "SHEETS"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   OLS Example    - OLS regression and prediction intervals on mtcars data"
    r = r + 1
    ws.Cells(r, 1).Value = "   Diagnostics    - 10 statistical diagnostic tests"
    r = r + 1
    ws.Cells(r, 1).Value = "   Regularized    - Ridge, Lasso, Elastic Net (edit Lambda/Alpha, click button)"
    r = r + 1
    ws.Cells(r, 1).Value = "   Cross Validation - K-Fold CV for OLS, Ridge, Lasso, Elastic Net"
    r = r + 1
    ws.Cells(r, 1).Value = "   Polynomial     - Fit polynomial curves (y = b0 + b1*x + b2*x^2 + ...)"
    r = r + 1
    ws.Cells(r, 1).Value = "   Instructions   - this page"
    r = r + 2

    ws.Cells(r, 1).Value = "READING OLS OUTPUT"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   Term          - coefficient name (Intercept, X1=cyl, X2=disp, X3=hp, X4=wt)"
    r = r + 1
    ws.Cells(r, 1).Value = "   Coefficient   - estimated value"
    r = r + 1
    ws.Cells(r, 1).Value = "   Std Error     - standard error of the estimate"
    r = r + 1
    ws.Cells(r, 1).Value = "   t Stat        - coefficient / std error"
    r = r + 1
    ws.Cells(r, 1).Value = "   p-Value       - two-tailed significance (< 0.05 is conventionally significant)"
    r = r + 1
    ws.Cells(r, 1).Value = "   R-squared            - proportion of variance explained (0-1)"
    r = r + 1
    ws.Cells(r, 1).Value = "   Adj R-squared        - R-squared adjusted for number of predictors"
    r = r + 1
    ws.Cells(r, 1).Value = "   F-stat / p(F) - overall model significance"
    r = r + 1
    ws.Cells(r, 1).Value = "   MSE           - mean squared error of residuals"
    r = r + 2

    ws.Cells(r, 1).Value = "READING DIAGNOSTIC OUTPUT"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   Statistic     - test statistic (chi-squared, F, t, or W depending on test)"
    r = r + 1
    ws.Cells(r, 1).Value = "   p-Value       - p < 0.05 suggests the assumption may be violated"
    r = r + 1
    ws.Cells(r, 1).Value = "   df            - degrees of freedom (where applicable)"
    r = r + 1
    ws.Cells(r, 1).Value = "   Durbin-Watson - rho column shows estimated autocorrelation (not a p-value)"
    r = r + 2

    ws.Cells(r, 1).Value = "REGULARIZED REGRESSION"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   Lambda        - regularization strength. Larger = more shrinkage."
    r = r + 1
    ws.Cells(r, 1).Value = "   Alpha         - mixing parameter. 0 = pure Ridge. 1 = pure Lasso. 0.5 = Elastic Net."
    r = r + 1
    ws.Cells(r, 1).Value = "   Non-zero      - number of non-zero coefficients after Lasso/Elastic Net shrinkage."
    r = r + 2

    ws.Cells(r, 1).Value = "K-FOLD CROSS VALIDATION"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   nFolds        - number of folds (default 5). More folds = less bias per fold, slower."
    r = r + 1
    ws.Cells(r, 1).Value = "   Mean MSE/RMSE - average error across all folds (lower is better)."
    r = r + 1
    ws.Cells(r, 1).Value = "   Std MSE/RMSE  - standard deviation of error across folds (lower = more stable)."
    r = r + 1
    ws.Cells(r, 1).Value = "   Mean R-squared       - average R-squared across folds (higher is better)."
    r = r + 2

    ws.Cells(r, 1).Value = "POLYNOMIAL REGRESSION"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   Degree        - highest power of x (1=linear, 2=quadratic, 3=cubic, etc.)"
    r = r + 1
    ws.Cells(r, 1).Value = "   Center        - subtract mean of x before creating powers (recommended for degree >= 3)"
    r = r + 1
    ws.Cells(r, 1).Value = "                   Reduces multicollinearity between x, x^2, x^3, etc."
    r = r + 2

    ws.Cells(r, 1).Value = "LIBRARY INFORMATION"
    ws.Cells(r, 1).Font.Bold = True
    r = r + 1
    ws.Cells(r, 1).Value = "   Name:     linreg-core"
    r = r + 1
    ws.Cells(r, 1).Value = "   Version:  " & LinReg_Version()
    r = r + 1
    ws.Cells(r, 1).Value = "   License:  MIT OR Apache-2.0"
    r = r + 1
    ws.Cells(r, 1).Value = "   Source:   https://github.com/jesse-anderson/linreg-core"

    ws.Columns(1).ColumnWidth = 80
End Sub

' ==============================================================================
' SECTION 5 - OLS sheet button macros
' ==============================================================================

Public Sub RunOLS()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_OLS)

    Dim result As Variant
    result = LinReg_OLS(YRange(), XRange())
    PasteResult ws, result, OLS_OUT_ROW, OLS_OUT_COL

    On Error Resume Next
    If UBound(result, 1) > 0 Then
        StyleHeader ws, OLS_OUT_ROW, OLS_OUT_COL, 5
        ws.Columns(OLS_OUT_COL).ColumnWidth = 14
        ws.Columns(OLS_OUT_COL + 1).ColumnWidth = 14
        ws.Columns(OLS_OUT_COL + 2).ColumnWidth = 12
        ws.Columns(OLS_OUT_COL + 3).ColumnWidth = 10
        ws.Columns(OLS_OUT_COL + 4).ColumnWidth = 10
    End If
    On Error GoTo 0
End Sub

Public Sub RunPredictionIntervals()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_OLS)

    ' Train on rows 1-15, predict rows 16-20
    Dim yTrain As Range, xTrain As Range, xNew As Range
    With ws
        Set yTrain = .Range(.Cells(DATA_R1,      Y_COL),   .Cells(DATA_R1 + 14, Y_COL))
        Set xTrain = .Range(.Cells(DATA_R1,      X_FIRST), .Cells(DATA_R1 + 14, X_LAST))
        Set xNew   = .Range(.Cells(DATA_R1 + 15, X_FIRST), .Cells(DATA_R2,      X_LAST))
    End With

    Dim result As Variant
    result = LinReg_PredictionIntervals(yTrain, xTrain, xNew, 0.05)

    Dim outRow As Long
    outRow = DATA_R2 + 3
    With ws.Cells(outRow - 1, Y_COL)
        .Value = "95% Prediction Intervals - rows 16-20 predicted from model trained on rows 1-15"
        .Font.Bold = True
    End With
    PasteResult ws, result, outRow, Y_COL

    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, outRow, Y_COL, 4
    On Error GoTo 0
End Sub

Public Sub RunVIF()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_OLS)

    Dim result As Variant
    result = LinReg_VIF(YRange(), XRange())

    Dim outRow As Long
    outRow = DATA_R2 + 10
    With ws.Cells(outRow - 1, OLS_OUT_COL)
        .Value = "VIF (Variance Inflation Factors)"
        .Font.Bold = True
    End With
    PasteResult ws, result, outRow, OLS_OUT_COL

    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, outRow, OLS_OUT_COL, 1
    On Error GoTo 0
End Sub

Public Sub RunWLS()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_OLS)

    ' Create simple weights: 1.0 for all observations (equal to OLS for demo)
    ' In practice, users would edit these based on heteroscedasticity patterns
    Dim weights As Range
    Set weights = ws.Range(ws.Cells(DATA_R1, X_LAST + 1), ws.Cells(DATA_R2, X_LAST + 1))
    weights.ClearContents
    Dim i As Long
    For i = 1 To weights.Count
        weights.Cells(i, 1).Value = 1#
    Next i
    ws.Cells(DATA_R1 - 1, X_LAST + 1).Value = "Weights"
    ws.Cells(DATA_R1 - 1, X_LAST + 1).Font.Bold = True

    Dim result As Variant
    result = LinReg_WLS(YRange(), XRange(), weights)

    Dim outRow As Long
    outRow = DATA_R2 + 15
    With ws.Cells(outRow - 1, OLS_OUT_COL)
        .Value = "WLS Results (weights = 1.0 for demo, edit column F for custom weights)"
        .Font.Bold = True
    End With
    PasteResult ws, result, outRow, OLS_OUT_COL

    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, outRow, OLS_OUT_COL, 5
    On Error GoTo 0
End Sub

' ==============================================================================
' SECTION 6 - Diagnostics sheet button macros
' ==============================================================================

Private Sub WriteDiagRow(ws As Worksheet, rowNum As Long, _
                          label As String, result As Variant)
    ws.Cells(rowNum, 1).Value = label
    If Not IsArray(result) Then
        ws.Cells(rowNum, 2).Value = CStr(result) : Exit Sub
    End If
    If UBound(result) = 0 Then
        ws.Cells(rowNum, 2).Value = "ERROR: " & CStr(result(0)) : Exit Sub
    End If
    ws.Cells(rowNum, 2).Value = result(0)
    ws.Cells(rowNum, 3).Value = result(1)
    ws.Cells(rowNum, 4).Value = result(2)
End Sub

Public Sub RunAllDiagnostics()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_DIAG)

    ws.Range("A3:D20").ClearContents
    ws.Range("A3:D20").ClearFormats

    ws.Cells(3, 1).Value = "Test"
    ws.Cells(3, 2).Value = "Statistic"
    ws.Cells(3, 3).Value = "p-Value / rho"
    ws.Cells(3, 4).Value = "df"
    StyleHeader ws, 3, 1, 4

    Dim y As Range, x As Range
    Set y = YRange()
    Set x = XRange()

    WriteDiagRow ws, 4,  "Breusch-Pagan",           LinReg_BreuschPagan(y, x)
    WriteDiagRow ws, 5,  "White",                    LinReg_White(y, x)
    WriteDiagRow ws, 6,  "Jarque-Bera",              LinReg_JarqueBera(y, x)
    WriteDiagRow ws, 7,  "Shapiro-Wilk",             LinReg_ShapiroWilk(y, x)
    WriteDiagRow ws, 8,  "Anderson-Darling",         LinReg_AndersonDarling(y, x)
    WriteDiagRow ws, 9,  "Harvey-Collier",           LinReg_HarveyCollier(y, x)
    WriteDiagRow ws, 10, "Rainbow (fraction = 0.5)", LinReg_Rainbow(y, x, 0.5)
    WriteDiagRow ws, 11, "RESET (powers 2, 3)",      LinReg_Reset(y, x)
    WriteDiagRow ws, 12, "Durbin-Watson",            LinReg_DurbinWatson(y, x)
    WriteDiagRow ws, 13, "Breusch-Godfrey (lag=1)",  LinReg_BreuschGodfrey(y, x, 1)

    MsgBox "Diagnostics complete.", vbInformation, "LinregCore"
End Sub

' ==============================================================================
' SECTION 7 - Regularized sheet button macros
' ==============================================================================

Private Function GetLambda() As Double
    GetLambda = CDbl(ThisWorkbook.Sheets(SH_REG).Range("B2").Value)
End Function

Private Function GetAlpha() As Double
    GetAlpha = CDbl(ThisWorkbook.Sheets(SH_REG).Range("B3").Value)
End Function

Public Sub RunRidge()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_REG)
    Dim result As Variant
    result = LinReg_Ridge(YRange(), XRange(), GetLambda(), True)
    PasteResult ws, result, 1, 4
    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, 1, 4, 2
    On Error GoTo 0
End Sub

Public Sub RunLasso()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_REG)
    Dim result As Variant
    result = LinReg_Lasso(YRange(), XRange(), GetLambda(), True)
    PasteResult ws, result, 1, 4
    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, 1, 4, 2
    On Error GoTo 0
End Sub

Public Sub RunElasticNet()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_REG)
    Dim result As Variant
    result = LinReg_ElasticNet(YRange(), XRange(), GetLambda(), GetAlpha(), True)
    PasteResult ws, result, 1, 4
    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, 1, 4, 2
    On Error GoTo 0
End Sub

' ==============================================================================
' SECTION 8 - Cross Validation sheet button macros
' ==============================================================================

Private Function GetNFolds() As Long
    GetNFolds = CLng(ThisWorkbook.Sheets(SH_CV).Range("B4").Value)
End Function

Private Sub WriteCVRow(ws As Worksheet, rowNum As Long, label As String, result As Variant)
    ws.Cells(rowNum, 1).Value = label
    If Not IsArray(result) Then
        ws.Cells(rowNum, 2).Value = CStr(result)
        Exit Sub
    End If
    If UBound(result) = 0 Then
        ws.Cells(rowNum, 2).Value = "ERROR: " & CStr(result(0))
        Exit Sub
    End If
    ' Result is 1x6: {nFolds, meanMSE, sdMSE, meanRMSE, sdRMSE, meanR2}
    ws.Cells(rowNum, 2).Value = result(0)  ' nFolds
    ws.Cells(rowNum, 3).Value = result(1)  ' meanMSE
    ws.Cells(rowNum, 4).Value = result(2)  ' sdMSE
    ws.Cells(rowNum, 5).Value = result(3)  ' meanRMSE
    ws.Cells(rowNum, 6).Value = result(4)  ' sdRMSE
    ws.Cells(rowNum, 7).Value = result(5)  ' meanR2
End Sub

Public Sub RunKFoldOLS()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_CV)

    ws.Range("A9:G20").ClearContents
    ws.Range("A9:G20").ClearFormats

    ws.Cells(9, 1).Value = "Method"
    ws.Cells(9, 2).Value = "nFolds"
    ws.Cells(9, 3).Value = "Mean MSE"
    ws.Cells(9, 4).Value = "Std MSE"
    ws.Cells(9, 5).Value = "Mean RMSE"
    ws.Cells(9, 6).Value = "Std RMSE"
    ws.Cells(9, 7).Value = "Mean R-squared"
    StyleHeader ws, 9, 1, 7

    WriteCVRow ws, 10, "OLS", LinReg_KFoldOLS(YRange(), XRange(), GetNFolds)
    MsgBox "Cross-validation complete.", vbInformation, "LinregCore"
End Sub

Public Sub RunKFoldRidge()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_CV)

    ws.Range("A9:G20").ClearContents
    ws.Range("A9:G20").ClearFormats

    ws.Cells(9, 1).Value = "Method"
    ws.Cells(9, 2).Value = "nFolds"
    ws.Cells(9, 3).Value = "Mean MSE"
    ws.Cells(9, 4).Value = "Std MSE"
    ws.Cells(9, 5).Value = "Mean RMSE"
    ws.Cells(9, 6).Value = "Std RMSE"
    ws.Cells(9, 7).Value = "Mean R-squared"
    StyleHeader ws, 9, 1, 7

    WriteCVRow ws, 10, "Ridge (lambda=" & GetLambda() & ")", LinReg_KFoldRidge(YRange(), XRange(), GetLambda(), True, GetNFolds)
    WriteCVRow ws, 11, "Ridge (lambda=0.1)", LinReg_KFoldRidge(YRange(), XRange(), 0.1, True, GetNFolds)
    WriteCVRow ws, 12, "Ridge (lambda=0.01)", LinReg_KFoldRidge(YRange(), XRange(), 0.01, True, GetNFolds)
    MsgBox "Cross-validation complete.", vbInformation, "LinregCore"
End Sub

Public Sub RunKFoldLasso()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_CV)

    ws.Range("A9:G20").ClearContents
    ws.Range("A9:G20").ClearFormats

    ws.Cells(9, 1).Value = "Method"
    ws.Cells(9, 2).Value = "nFolds"
    ws.Cells(9, 3).Value = "Mean MSE"
    ws.Cells(9, 4).Value = "Std MSE"
    ws.Cells(9, 5).Value = "Mean RMSE"
    ws.Cells(9, 6).Value = "Std RMSE"
    ws.Cells(9, 7).Value = "Mean R-squared"
    StyleHeader ws, 9, 1, 7

    WriteCVRow ws, 10, "Lasso (lambda=" & GetLambda() & ")", LinReg_KFoldLasso(YRange(), XRange(), GetLambda(), True, GetNFolds)
    WriteCVRow ws, 11, "Lasso (lambda=0.1)", LinReg_KFoldLasso(YRange(), XRange(), 0.1, True, GetNFolds)
    MsgBox "Cross-validation complete.", vbInformation, "LinregCore"
End Sub

Public Sub RunKFoldElasticNet()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_CV)

    ws.Range("A9:G20").ClearContents
    ws.Range("A9:G20").ClearFormats

    ws.Cells(9, 1).Value = "Method"
    ws.Cells(9, 2).Value = "nFolds"
    ws.Cells(9, 3).Value = "Mean MSE"
    ws.Cells(9, 4).Value = "Std MSE"
    ws.Cells(9, 5).Value = "Mean RMSE"
    ws.Cells(9, 6).Value = "Std RMSE"
    ws.Cells(9, 7).Value = "Mean R-squared"
    StyleHeader ws, 9, 1, 7

    WriteCVRow ws, 10, "Elastic Net", LinReg_KFoldElasticNet(YRange(), XRange(), GetLambda(), GetAlpha(), True, GetNFolds)
    MsgBox "Cross-validation complete.", vbInformation, "LinregCore"
End Sub

' ==============================================================================
' SECTION 9 - Polynomial sheet button macros
' ==============================================================================

Public Sub RunPolynomial()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_POLY)

    Dim degree As Long
    degree = CLng(ws.Range("B27").Value)

    Dim center As Boolean
    center = (ws.Range("B28").Value = True)

    ' Data from polynomial sheet (rows 5-24)
    Dim y As Range, x As Range
    Set y = ws.Range(ws.Cells(5, 2), ws.Cells(24, 2))
    Set x = ws.Range(ws.Cells(5, 1), ws.Cells(24, 1))

    Dim result As Variant
    result = LinReg_Polynomial(y, x, degree, center)

    Dim outRow As Long
    outRow = 30
    PasteResult ws, result, outRow, 4

    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, outRow, 4, 2
    On Error GoTo 0
End Sub

Public Sub RunPolynomialQuad()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_POLY)

    Dim y As Range, x As Range
    Set y = ws.Range(ws.Cells(5, 2), ws.Cells(24, 2))
    Set x = ws.Range(ws.Cells(5, 1), ws.Cells(24, 1))

    Dim result As Variant
    result = LinReg_Polynomial(y, x, 2, False)

    Dim outRow As Long
    outRow = 30
    PasteResult ws, result, outRow, 4

    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, outRow, 4, 2
    On Error GoTo 0
End Sub

Public Sub RunPolynomialCubic()
    Dim ws As Worksheet
    Set ws = ThisWorkbook.Sheets(SH_POLY)

    Dim y As Range, x As Range
    Set y = ws.Range(ws.Cells(5, 2), ws.Cells(24, 2))
    Set x = ws.Range(ws.Cells(5, 1), ws.Cells(24, 1))

    Dim result As Variant
    result = LinReg_Polynomial(y, x, 3, True)  ' Center recommended for cubic

    Dim outRow As Long
    outRow = 30
    PasteResult ws, result, outRow, 4

    On Error Resume Next
    If UBound(result, 1) > 0 Then StyleHeader ws, outRow, 4, 2
    On Error GoTo 0
End Sub
