Attribute VB_Name = "LinregCore"
' ==============================================================================
' LinregCore.bas  -  VBA/Excel bindings for linreg_core.dll
' Requires Office 2010+ (VBA7).  Pre-VBA7 Declares are included for
' reference but all high-level wrapper functions require VBA7.
' ==============================================================================
'
' QUICK START
'   1. Copy linreg_core_x64.dll (64-bit Excel) OR linreg_core_x86.dll
'      (32-bit Excel) into the same folder as your workbook.
'   2. VBA Editor (Alt+F11): File > Import File > select LinregCore.bas
'   3. In ThisWorkbook module, add:
'          Private Sub Workbook_Open()
'              LR_Init
'          End Sub
'   4. Call any wrapper function from VBA code or a macro button:
'          Dim tbl As Variant
'          tbl = LinReg_OLS(Range("B2:B101"), Range("C2:E101"))
'          Range("H2").Resize(UBound(tbl,1)+1, UBound(tbl,2)+1).Value = tbl
'
' WRAPPER FUNCTION REFERENCE
'   LinReg_OLS(y, X)                               (k+6)x5 coef + stats table
'   LinReg_WLS(y, X, w)                            (k+6)x5 coef + stats table (weighted)
'   LinReg_Ridge(y, X, lambda [,std])              (p+8)x2 coef + stats table
'   LinReg_Lasso(y, X, lambda [,std,iter,tol])     (p+9)x2 coef + stats table
'   LinReg_ElasticNet(y, X, lambda, alpha [,...])  (p+10)x2 coef + stats table
'   LinReg_PredictionIntervals(y,X,newX [,alpha])  (n_new+1)x4 PI table
'   LinReg_BreuschPagan(y, X)                      1x3: {stat, p-value, df}
'   LinReg_White(y, X)                             1x3: {stat, p-value, df}
'   LinReg_JarqueBera(y, X)                        1x3: {stat, p-value, df}
'   LinReg_ShapiroWilk(y, X)                       1x3: {stat, p-value, df}
'   LinReg_AndersonDarling(y, X)                   1x3: {stat, p-value, df}
'   LinReg_HarveyCollier(y, X)                     1x3: {stat, p-value, df}
'   LinReg_Rainbow(y, X [,fraction])               1x3: {stat, p-value, df}
'   LinReg_Reset(y, X)                             1x3: {stat, p-value, df}
'   LinReg_DurbinWatson(y, X)                      1x3: {DW stat, rho, ""}
'   LinReg_BreuschGodfrey(y, X [,lagOrder])        1x3: {stat, p-value, df}
'   LinReg_VIF(y, X)                               px1 array of VIF values per predictor
'   LinReg_CooksDistance(y, X)                     nx1 Cook's distance per observation
'   LinReg_DFFITS(y, X)                            nx1 DFFITS value per observation
'   LinReg_DFBETAS(y, X)                           (n+1)x(p+1): header row + DFBETAS matrix
'   LinReg_LambdaPath(y, X [,nLambda,lmr,alpha])  Lx1 lambda sequence
'   LinReg_KFoldOLS(y, X [,nFolds])               1x6: {nFolds,meanMSE,sdMSE,meanRMSE,sdRMSE,meanR2}
'   LinReg_KFoldRidge(y, X, lambda [,std,nF])     1x6: same CV metrics
'   LinReg_KFoldLasso(y, X, lambda [,std,nF])     1x6: same CV metrics
'   LinReg_KFoldElasticNet(y,X,lam,alpha [,...])  1x6: same CV metrics
'   LinReg_Version()                               version string (e.g. "0.8.0")
'
' ERROR HANDLING
'   All wrappers return a 1-element array containing an error message string
'   if the DLL call fails.  Check: If IsArray(result) And UBound(result)=0 Then
'
' ==============================================================================
Option Explicit

' ==============================================================================
' SECTION 1 - Private DLL Declarations
' ==============================================================================

#If VBA7 Then
    ' == kernel32: explicit DLL pre-load =======================================
    Private Declare PtrSafe Function LoadLibraryA Lib "kernel32" _
        (ByVal lpLibFileName As String) As LongPtr

    #If Win64 Then
        ' ======================================================================
        ' 64-bit Office (VBA7 + Win64)  -  linreg_core_x64.dll
        ' ======================================================================

        ' == Handle management =================================================
        Private Declare PtrSafe Function LR_OLS Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Sub LR_Free Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr)

        Private Declare PtrSafe Function LR_GetLastError Lib "linreg_core_x64.dll" _
            (ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == Scalar OLS getters ================================================
        Private Declare PtrSafe Function LR_GetRSquared Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetAdjRSquared Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetFStatistic Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetFPValue Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetMSE Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetNumCoefficients Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetNumObservations Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        ' == Vector OLS getters ================================================
        Private Declare PtrSafe Function LR_GetCoefficients Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetStdErrors Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetTStats Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetPValues Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetResiduals Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetFittedValues Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == Regularized regression ============================================
        Private Declare PtrSafe Function LR_Ridge Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long) As LongPtr

        Private Declare PtrSafe Function LR_Lasso Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long, _
             ByVal max_iter As Long, ByVal tol As Double) As LongPtr

        Private Declare PtrSafe Function LR_ElasticNet Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal alpha As Double, _
             ByVal standardize As Long, ByVal max_iter As Long, _
             ByVal tol As Double) As LongPtr

        Private Declare PtrSafe Function LR_GetIntercept Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetDF Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetNNonzero Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetConverged Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        ' == Diagnostic scalar getters =========================================
        Private Declare PtrSafe Function LR_GetStatistic Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetPValue Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetTestDF Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetAutocorrelation Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        ' == Diagnostic fit functions ==========================================
        Private Declare PtrSafe Function LR_BreuschPagan Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_JarqueBera Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_ShapiroWilk Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_AndersonDarling Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_HarveyCollier Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_White Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_Rainbow Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal fraction As Double) As LongPtr

        Private Declare PtrSafe Function LR_Reset Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal powers_ptr As LongPtr, ByVal powers_len As Long) As LongPtr

        Private Declare PtrSafe Function LR_DurbinWatson Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_BreuschGodfrey Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lag_order As Long) As LongPtr

        ' == Prediction intervals ==============================================
        Private Declare PtrSafe Function LR_PredictionIntervals Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal new_x_ptr As LongPtr, ByVal n_new As Long, _
             ByVal alpha As Double) As LongPtr

        Private Declare PtrSafe Function LR_GetPredicted Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetLowerBound Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetUpperBound Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetSEPred Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == Influence diagnostics =============================================
        Private Declare PtrSafe Function LR_CooksDistance Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_DFFITS Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_VIF Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_DFBETAS Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        ' == Generic vector / matrix getters ==================================
        Private Declare PtrSafe Function LR_GetVectorLength Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetVector Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetMatrixRows Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetMatrixCols Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetMatrix Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == WLS and lambda path ===============================================
        Private Declare PtrSafe Function LR_WLS Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal w_ptr As LongPtr) As LongPtr

        Private Declare PtrSafe Function LR_LambdaPath Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal nlambda As Long, ByVal lambda_min_ratio As Double, _
             ByVal alpha As Double) As LongPtr

        ' == K-Fold cross validation ===========================================
        Private Declare PtrSafe Function LR_KFoldOLS Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_KFoldRidge Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long, _
             ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_KFoldLasso Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long, _
             ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_KFoldElasticNet Lib "linreg_core_x64.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal alpha As Double, _
             ByVal standardize As Long, ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_GetCVNFolds Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetCVMeanMSE Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVStdMSE Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVMeanRMSE Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVStdRMSE Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVMeanR2 Lib "linreg_core_x64.dll" _
            (ByVal handle As LongPtr) As Double

        ' == Utilities =========================================================
        Private Declare PtrSafe Function LR_Version Lib "linreg_core_x64.dll" _
            (ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' Aliased so we can have a public Sub LR_Init() without name clash
        Private Declare PtrSafe Function LR_InitDLL Lib "linreg_core_x64.dll" _
            Alias "LR_Init" () As Long

    #Else
        ' ======================================================================
        ' 32-bit Office, VBA7  -  linreg_core_x86.dll
        ' (LongPtr is 4 bytes here, same width as Long)
        ' ======================================================================

        ' == Handle management =================================================
        Private Declare PtrSafe Function LR_OLS Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Sub LR_Free Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr)

        Private Declare PtrSafe Function LR_GetLastError Lib "linreg_core_x86.dll" _
            (ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == Scalar OLS getters ================================================
        Private Declare PtrSafe Function LR_GetRSquared Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetAdjRSquared Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetFStatistic Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetFPValue Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetMSE Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetNumCoefficients Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetNumObservations Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        ' == Vector OLS getters ================================================
        Private Declare PtrSafe Function LR_GetCoefficients Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetStdErrors Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetTStats Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetPValues Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetResiduals Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetFittedValues Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == Regularized regression ============================================
        Private Declare PtrSafe Function LR_Ridge Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long) As LongPtr

        Private Declare PtrSafe Function LR_Lasso Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long, _
             ByVal max_iter As Long, ByVal tol As Double) As LongPtr

        Private Declare PtrSafe Function LR_ElasticNet Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal alpha As Double, _
             ByVal standardize As Long, ByVal max_iter As Long, _
             ByVal tol As Double) As LongPtr

        Private Declare PtrSafe Function LR_GetIntercept Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetDF Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetNNonzero Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetConverged Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        ' == Diagnostic scalar getters =========================================
        Private Declare PtrSafe Function LR_GetStatistic Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetPValue Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetTestDF Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetAutocorrelation Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        ' == Diagnostic fit functions ==========================================
        Private Declare PtrSafe Function LR_BreuschPagan Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_JarqueBera Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_ShapiroWilk Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_AndersonDarling Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_HarveyCollier Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_White Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_Rainbow Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal fraction As Double) As LongPtr

        Private Declare PtrSafe Function LR_Reset Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal powers_ptr As LongPtr, ByVal powers_len As Long) As LongPtr

        Private Declare PtrSafe Function LR_DurbinWatson Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_BreuschGodfrey Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lag_order As Long) As LongPtr

        ' == Prediction intervals ==============================================
        Private Declare PtrSafe Function LR_PredictionIntervals Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal new_x_ptr As LongPtr, ByVal n_new As Long, _
             ByVal alpha As Double) As LongPtr

        Private Declare PtrSafe Function LR_GetPredicted Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetLowerBound Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetUpperBound Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetSEPred Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == Influence diagnostics =============================================
        Private Declare PtrSafe Function LR_CooksDistance Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_DFFITS Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_VIF Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        Private Declare PtrSafe Function LR_DFBETAS Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long) As LongPtr

        ' == Generic vector / matrix getters ==================================
        Private Declare PtrSafe Function LR_GetVectorLength Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetVector Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_GetMatrixRows Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetMatrixCols Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetMatrix Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr, ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        ' == WLS and lambda path ===============================================
        Private Declare PtrSafe Function LR_WLS Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal w_ptr As LongPtr) As LongPtr

        Private Declare PtrSafe Function LR_LambdaPath Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal nlambda As Long, ByVal lambda_min_ratio As Double, _
             ByVal alpha As Double) As LongPtr

        ' == K-Fold cross validation ===========================================
        Private Declare PtrSafe Function LR_KFoldOLS Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_KFoldRidge Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long, _
             ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_KFoldLasso Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal standardize As Long, _
             ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_KFoldElasticNet Lib "linreg_core_x86.dll" _
            (ByVal y_ptr As LongPtr, ByVal n As Long, _
             ByVal x_ptr As LongPtr, ByVal p As Long, _
             ByVal lambda As Double, ByVal alpha As Double, _
             ByVal standardize As Long, ByVal n_folds As Long) As LongPtr

        Private Declare PtrSafe Function LR_GetCVNFolds Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Long

        Private Declare PtrSafe Function LR_GetCVMeanMSE Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVStdMSE Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVMeanRMSE Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVStdRMSE Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        Private Declare PtrSafe Function LR_GetCVMeanR2 Lib "linreg_core_x86.dll" _
            (ByVal handle As LongPtr) As Double

        ' == Utilities =========================================================
        Private Declare PtrSafe Function LR_Version Lib "linreg_core_x86.dll" _
            (ByVal out_ptr As LongPtr, ByVal out_len As Long) As Long

        Private Declare PtrSafe Function LR_InitDLL Lib "linreg_core_x86.dll" _
            Alias "LR_Init" () As Long

    #End If  ' Win64

#Else
    ' ==========================================================================
    ' Pre-VBA7 (Office 2007 and earlier) - 32-bit only, linreg_core_x86.dll
    ' High-level wrapper functions below require VBA7; raw DLL calls work here.
    ' ==========================================================================
    Private Declare Function LoadLibraryA Lib "kernel32" _
        (ByVal lpLibFileName As String) As Long

    Private Declare Function LR_OLS Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Sub LR_Free Lib "linreg_core_x86.dll" _
        (ByVal handle As Long)

    Private Declare Function LR_GetLastError Lib "linreg_core_x86.dll" _
        (ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetRSquared Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetAdjRSquared Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetFStatistic Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetFPValue Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetMSE Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetNumCoefficients Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetNumObservations Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetCoefficients Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetStdErrors Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetTStats Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetPValues Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetResiduals Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetFittedValues Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_Ridge Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lambda As Double, ByVal standardize As Long) As Long

    Private Declare Function LR_Lasso Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lambda As Double, ByVal standardize As Long, _
         ByVal max_iter As Long, ByVal tol As Double) As Long

    Private Declare Function LR_ElasticNet Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lambda As Double, ByVal alpha As Double, _
         ByVal standardize As Long, ByVal max_iter As Long, _
         ByVal tol As Double) As Long

    Private Declare Function LR_GetIntercept Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetDF Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetNNonzero Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetConverged Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetStatistic Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetPValue Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetTestDF Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetAutocorrelation Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_BreuschPagan Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_JarqueBera Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_ShapiroWilk Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_AndersonDarling Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_HarveyCollier Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_White Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_Rainbow Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal fraction As Double) As Long

    Private Declare Function LR_Reset Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal powers_ptr As Long, ByVal powers_len As Long) As Long

    Private Declare Function LR_DurbinWatson Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_BreuschGodfrey Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lag_order As Long) As Long

    Private Declare Function LR_PredictionIntervals Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal new_x_ptr As Long, ByVal n_new As Long, _
         ByVal alpha As Double) As Long

    Private Declare Function LR_GetPredicted Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetLowerBound Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetUpperBound Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetSEPred Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_CooksDistance Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_DFFITS Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_VIF Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_DFBETAS Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long) As Long

    Private Declare Function LR_GetVectorLength Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetVector Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_GetMatrixRows Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetMatrixCols Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetMatrix Lib "linreg_core_x86.dll" _
        (ByVal handle As Long, ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_WLS Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal w_ptr As Long) As Long

    Private Declare Function LR_LambdaPath Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal nlambda As Long, ByVal lambda_min_ratio As Double, _
         ByVal alpha As Double) As Long

    Private Declare Function LR_KFoldOLS Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal n_folds As Long) As Long

    Private Declare Function LR_KFoldRidge Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lambda As Double, ByVal standardize As Long, _
         ByVal n_folds As Long) As Long

    Private Declare Function LR_KFoldLasso Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lambda As Double, ByVal standardize As Long, _
         ByVal n_folds As Long) As Long

    Private Declare Function LR_KFoldElasticNet Lib "linreg_core_x86.dll" _
        (ByVal y_ptr As Long, ByVal n As Long, _
         ByVal x_ptr As Long, ByVal p As Long, _
         ByVal lambda As Double, ByVal alpha As Double, _
         ByVal standardize As Long, ByVal n_folds As Long) As Long

    Private Declare Function LR_GetCVNFolds Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Long

    Private Declare Function LR_GetCVMeanMSE Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetCVStdMSE Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetCVMeanRMSE Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetCVStdRMSE Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_GetCVMeanR2 Lib "linreg_core_x86.dll" _
        (ByVal handle As Long) As Double

    Private Declare Function LR_Version Lib "linreg_core_x86.dll" _
        (ByVal out_ptr As Long, ByVal out_len As Long) As Long

    Private Declare Function LR_InitDLL Lib "linreg_core_x86.dll" _
        Alias "LR_Init" () As Long

#End If  ' VBA7

' ==============================================================================
' SECTION 2 - Initialisation
' ==============================================================================

' Call once from Workbook_Open (or manually before any other LinReg_* call).
' Pre-loads the correct DLL from the workbook's folder so that the Declare
' statements above can find it by name.
'
' Example in ThisWorkbook:
'   Private Sub Workbook_Open()
'       LR_Init
'   End Sub
Public Sub LR_Init()
    #If Win64 Then
        LoadLibraryA ThisWorkbook.Path & "\linreg_core_x64.dll"
    #Else
        LoadLibraryA ThisWorkbook.Path & "\linreg_core_x86.dll"
    #End If
End Sub

' ==============================================================================
' SECTION 3 - Private helpers
' ==============================================================================

' Read the last error string set by the DLL.
Private Function GetLastErrorMsg() As String
    Dim buf(0 To 511) As Byte
    Dim written As Long
    written = LR_GetLastError(VarPtr(buf(0)), 512)
    If written > 0 Then
        GetLastErrorMsg = Left$(StrConv(buf, vbUnicode), written)
    Else
        GetLastErrorMsg = "(unknown error)"
    End If
End Function

' Convert a single-column or single-row Range to a 0-based 1D Double array.
' Cells are read in worksheet order (top-to-bottom for a column,
' left-to-right for a row).
Private Function RangeToDoubleArray(rng As Range) As Double()
    Dim arr() As Double
    Dim n As Long
    n = rng.Cells.Count
    ReDim arr(0 To n - 1)
    Dim i As Long
    Dim c As Range
    i = 0
    For Each c In rng.Cells
        arr(i) = CDbl(c.Value)
        i = i + 1
    Next c
    RangeToDoubleArray = arr
End Function

' Convert a multi-column Range to a flat 0-based 1D Double array in
' row-major order (row 1 col 1, row 1 col 2, ..., row 2 col 1, ...).
' This matches the layout the DLL expects for predictor matrices.
Private Function RangeToMatrix(rng As Range) As Double()
    Dim arr() As Double
    Dim nRows As Long, nCols As Long
    Dim r As Long, c As Long
    nRows = rng.Rows.Count
    nCols = rng.Columns.Count
    ReDim arr(0 To nRows * nCols - 1)
    Dim idx As Long
    idx = 0
    For r = 1 To nRows
        For c = 1 To nCols
            arr(idx) = CDbl(rng.Cells(r, c).Value)
            idx = idx + 1
        Next c
    Next r
    RangeToMatrix = arr
End Function

' Shared extractor for all standard diagnostic handles.
' Returns a 1x3 Variant array: {statistic, p-value, df}.
' On error returns a 1-element array with the error message string.
#If VBA7 Then
Private Function DiagResult(h As LongPtr) As Variant
    If h = 0 Then
        DiagResult = Array(GetLastErrorMsg())
        Exit Function
    End If
    Dim result(0 To 2) As Variant
    result(0) = LR_GetStatistic(h)
    result(1) = LR_GetPValue(h)
    result(2) = LR_GetTestDF(h)
    LR_Free h
    DiagResult = result
End Function
#End If

' ==============================================================================
' SECTION 4 - OLS regression wrapper
' ==============================================================================

' Fit an OLS regression model and return a formatted summary table.
'
' Parameters:
'   yRange  - single column (or row) of n response values
'   xRange  - n rows x p columns of predictor values (NO intercept column)
'
' Returns a (k+6) x 5 Variant array (k = p+1 including intercept):
'   Row 0:       header  ["Term", "Coefficient", "Std Error", "t Stat", "p-Value"]
'   Rows 1..k:   one row per coefficient
'   Row k+1:     blank separator
'   Row k+2:     ["R-squared",     r_squared,    "", "",       ""]
'   Row k+3:     ["Adj R-squared", adj_r_squared,"", "",       ""]
'   Row k+4:     ["F-stat", f_statistic,  "p(F)", f_p_value, ""]
'   Row k+5:     ["MSE",    mse,          "n",  n_obs,   ""]
'
' On error returns a 1-element array containing the error message string.
#If VBA7 Then
Public Function LinReg_OLS(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    Dim h As LongPtr

    n = yRange.Cells.Count
    p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange)
    X = RangeToMatrix(xRange)

    h = LR_OLS(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p))
    If h = 0 Then
        LinReg_OLS = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim k As Long
    k = LR_GetNumCoefficients(h)  ' intercept + p slopes

    Dim coefs() As Double, ses() As Double
    Dim tstats() As Double, pvals() As Double
    ReDim coefs(0 To k - 1)
    ReDim ses(0 To k - 1)
    ReDim tstats(0 To k - 1)
    ReDim pvals(0 To k - 1)

    LR_GetCoefficients h, VarPtr(coefs(0)), k
    LR_GetStdErrors    h, VarPtr(ses(0)),   k
    LR_GetTStats       h, VarPtr(tstats(0)), k
    LR_GetPValues      h, VarPtr(pvals(0)),  k

    Dim r2 As Double, adjr2 As Double
    Dim fstat As Double, fp As Double, mse As Double
    r2    = LR_GetRSquared(h)
    adjr2 = LR_GetAdjRSquared(h)
    fstat = LR_GetFStatistic(h)
    fp    = LR_GetFPValue(h)
    mse   = LR_GetMSE(h)
    n     = LR_GetNumObservations(h)

    LR_Free h

    ' Build result table: header + k coef rows + blank + 4 stats rows
    Dim nRows As Long
    nRows = k + 6
    Dim result() As Variant
    ReDim result(0 To nRows - 1, 0 To 4)

    ' Header
    result(0, 0) = "Term"
    result(0, 1) = "Coefficient"
    result(0, 2) = "Std Error"
    result(0, 3) = "t Stat"
    result(0, 4) = "p-Value"

    ' Coefficient rows
    Dim i As Long
    For i = 0 To k - 1
        result(i + 1, 0) = IIf(i = 0, "Intercept", "X" & i)
        result(i + 1, 1) = coefs(i)
        result(i + 1, 2) = ses(i)
        result(i + 1, 3) = tstats(i)
        result(i + 1, 4) = pvals(i)
    Next i

    ' Blank separator
    ' (row k+1 is already empty from ReDim)

    ' Model statistics
    result(k + 2, 0) = "R-squared"
    result(k + 2, 1) = r2
    result(k + 3, 0) = "Adj R-squared"
    result(k + 3, 1) = adjr2
    result(k + 4, 0) = "F-stat"
    result(k + 4, 1) = fstat
    result(k + 4, 2) = "p(F)"
    result(k + 4, 3) = fp
    result(k + 5, 0) = "MSE"
    result(k + 5, 1) = mse
    result(k + 5, 2) = "n"
    result(k + 5, 3) = n

    LinReg_OLS = result
End Function
#End If

' ==============================================================================
' SECTION 5 - Regularized regression wrappers
' ==============================================================================

' Shared layout for Ridge / Lasso / Elastic Net result tables:
'   Row 0:       header  ["Term", "Coefficient"]
'   Row 1:       ["Intercept", intercept]
'   Rows 2..p+1: ["X1"..."Xp", coef]
'   Row p+2:     blank
'   Row p+3:     ["R-squared",      r_squared]
'   Row p+4:     ["Adj R-squared",  adj_r_squared]
'   Row p+5:     ["MSE",     mse]
'   Row p+6:     model-specific stat (lambda, eff. df, n_nonzero, converged)
'   Row p+7+:    additional model-specific stats
'
' On error returns a 1-element array containing the error message string.

#If VBA7 Then

' == Ridge ======================================================================
' standardize: True (default) = standardize predictors before fitting (glmnet style)
Public Function LinReg_Ridge(yRange As Range, xRange As Range, _
                              lambda As Double, _
                              Optional standardize As Boolean = True) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    Dim h As LongPtr

    n = yRange.Cells.Count
    p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange)
    X = RangeToMatrix(xRange)

    h = LR_Ridge(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                 lambda, IIf(standardize, 1, 0))
    If h = 0 Then
        LinReg_Ridge = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim intercept As Double
    intercept = LR_GetIntercept(h)

    Dim nCoef As Long
    nCoef = LR_GetNumCoefficients(h)  ' slopes only (no intercept in this count)

    Dim coefs() As Double
    ReDim coefs(0 To nCoef - 1)
    LR_GetCoefficients h, VarPtr(coefs(0)), nCoef

    Dim r2 As Double, adjr2 As Double, mse As Double, effDF As Double
    r2    = LR_GetRSquared(h)
    adjr2 = LR_GetAdjRSquared(h)
    mse   = LR_GetMSE(h)
    effDF = LR_GetDF(h)

    LR_Free h

    Dim nRows As Long
    nRows = nCoef + 8  ' header + intercept + slopes + blank + 5 stats
    Dim result() As Variant
    ReDim result(0 To nRows - 1, 0 To 1)

    result(0, 0) = "Term"
    result(0, 1) = "Coefficient"
    result(1, 0) = "Intercept"
    result(1, 1) = intercept

    Dim i As Long
    For i = 0 To nCoef - 1
        result(i + 2, 0) = "X" & (i + 1)
        result(i + 2, 1) = coefs(i)
    Next i

    Dim sep As Long
    sep = nCoef + 2  ' blank row index
    ' row sep is already empty

    result(sep + 1, 0) = "R-squared"
    result(sep + 1, 1) = r2
    result(sep + 2, 0) = "Adj R-squared"
    result(sep + 2, 1) = adjr2
    result(sep + 3, 0) = "MSE"
    result(sep + 3, 1) = mse
    result(sep + 4, 0) = "Lambda"
    result(sep + 4, 1) = lambda
    result(sep + 5, 0) = "Eff. df"
    result(sep + 5, 1) = effDF

    LinReg_Ridge = result
End Function

' == Lasso ======================================================================
Public Function LinReg_Lasso(yRange As Range, xRange As Range, _
                              lambda As Double, _
                              Optional standardize As Boolean = True, _
                              Optional maxIter As Long = 10000, _
                              Optional tol As Double = 0.0000001) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    Dim h As LongPtr

    n = yRange.Cells.Count
    p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange)
    X = RangeToMatrix(xRange)

    h = LR_Lasso(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                 lambda, IIf(standardize, 1, 0), maxIter, tol)
    If h = 0 Then
        LinReg_Lasso = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim intercept As Double
    intercept = LR_GetIntercept(h)

    Dim nCoef As Long
    nCoef = LR_GetNumCoefficients(h)

    Dim coefs() As Double
    ReDim coefs(0 To nCoef - 1)
    LR_GetCoefficients h, VarPtr(coefs(0)), nCoef

    Dim r2 As Double, adjr2 As Double, mse As Double
    Dim nNonzero As Long, converged As Long
    r2        = LR_GetRSquared(h)
    adjr2     = LR_GetAdjRSquared(h)
    mse       = LR_GetMSE(h)
    nNonzero  = LR_GetNNonzero(h)
    converged = LR_GetConverged(h)

    LR_Free h

    Dim nRows As Long
    nRows = nCoef + 9
    Dim result() As Variant
    ReDim result(0 To nRows - 1, 0 To 1)

    result(0, 0) = "Term"
    result(0, 1) = "Coefficient"
    result(1, 0) = "Intercept"
    result(1, 1) = intercept

    Dim i As Long
    For i = 0 To nCoef - 1
        result(i + 2, 0) = "X" & (i + 1)
        result(i + 2, 1) = coefs(i)
    Next i

    Dim sep As Long
    sep = nCoef + 2

    result(sep + 1, 0) = "R-squared"
    result(sep + 1, 1) = r2
    result(sep + 2, 0) = "Adj R-squared"
    result(sep + 2, 1) = adjr2
    result(sep + 3, 0) = "MSE"
    result(sep + 3, 1) = mse
    result(sep + 4, 0) = "Lambda"
    result(sep + 4, 1) = lambda
    result(sep + 5, 0) = "Non-zero"
    result(sep + 5, 1) = nNonzero
    result(sep + 6, 0) = "Converged"
    result(sep + 6, 1) = IIf(converged = 1, "Yes", "No")

    LinReg_Lasso = result
End Function

' == Elastic Net ================================================================
' alpha: 0 = Ridge, 1 = Lasso, 0.5 = equal mix
Public Function LinReg_ElasticNet(yRange As Range, xRange As Range, _
                                   lambda As Double, alpha As Double, _
                                   Optional standardize As Boolean = True, _
                                   Optional maxIter As Long = 10000, _
                                   Optional tol As Double = 0.0000001) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    Dim h As LongPtr

    n = yRange.Cells.Count
    p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange)
    X = RangeToMatrix(xRange)

    h = LR_ElasticNet(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                      lambda, alpha, IIf(standardize, 1, 0), maxIter, tol)
    If h = 0 Then
        LinReg_ElasticNet = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim intercept As Double
    intercept = LR_GetIntercept(h)

    Dim nCoef As Long
    nCoef = LR_GetNumCoefficients(h)

    Dim coefs() As Double
    ReDim coefs(0 To nCoef - 1)
    LR_GetCoefficients h, VarPtr(coefs(0)), nCoef

    Dim r2 As Double, adjr2 As Double, mse As Double
    Dim nNonzero As Long, converged As Long
    r2        = LR_GetRSquared(h)
    adjr2     = LR_GetAdjRSquared(h)
    mse       = LR_GetMSE(h)
    nNonzero  = LR_GetNNonzero(h)
    converged = LR_GetConverged(h)

    LR_Free h

    Dim nRows As Long
    nRows = nCoef + 10
    Dim result() As Variant
    ReDim result(0 To nRows - 1, 0 To 1)

    result(0, 0) = "Term"
    result(0, 1) = "Coefficient"
    result(1, 0) = "Intercept"
    result(1, 1) = intercept

    Dim i As Long
    For i = 0 To nCoef - 1
        result(i + 2, 0) = "X" & (i + 1)
        result(i + 2, 1) = coefs(i)
    Next i

    Dim sep As Long
    sep = nCoef + 2

    result(sep + 1, 0) = "R-squared"
    result(sep + 1, 1) = r2
    result(sep + 2, 0) = "Adj R-squared"
    result(sep + 2, 1) = adjr2
    result(sep + 3, 0) = "MSE"
    result(sep + 3, 1) = mse
    result(sep + 4, 0) = "Lambda"
    result(sep + 4, 1) = lambda
    result(sep + 5, 0) = "Alpha"
    result(sep + 5, 1) = alpha
    result(sep + 6, 0) = "Non-zero"
    result(sep + 6, 1) = nNonzero
    result(sep + 7, 0) = "Converged"
    result(sep + 7, 1) = IIf(converged = 1, "Yes", "No")

    LinReg_ElasticNet = result
End Function

#End If  ' VBA7 (regularized section)

' ==============================================================================
' SECTION 6 - Diagnostic test wrappers
' ==============================================================================
' All diagnostics return a 1x3 Variant: {statistic, p-value, degrees-of-freedom}
' Durbin-Watson returns:                {DW statistic, rho (=1-DW/2), ""}
' On error: 1-element array with error message string.
' ==============================================================================

#If VBA7 Then

Public Function LinReg_BreuschPagan(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_BreuschPagan = DiagResult( _
        LR_BreuschPagan(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p)))
End Function

Public Function LinReg_White(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_White = DiagResult( _
        LR_White(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p)))
End Function

Public Function LinReg_JarqueBera(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_JarqueBera = DiagResult( _
        LR_JarqueBera(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p)))
End Function

Public Function LinReg_ShapiroWilk(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_ShapiroWilk = DiagResult( _
        LR_ShapiroWilk(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p)))
End Function

Public Function LinReg_AndersonDarling(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_AndersonDarling = DiagResult( _
        LR_AndersonDarling(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p)))
End Function

Public Function LinReg_HarveyCollier(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_HarveyCollier = DiagResult( _
        LR_HarveyCollier(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p)))
End Function

' fraction: proportion of central observations used (default 0.5)
Public Function LinReg_Rainbow(yRange As Range, xRange As Range, _
                                Optional fraction As Double = 0.5) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_Rainbow = DiagResult( _
        LR_Rainbow(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), fraction))
End Function

' Uses default powers {2, 3}.  For custom powers call LR_Reset directly.
Public Function LinReg_Reset(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    Dim powers(0 To 1) As Long
    powers(0) = 2 : powers(1) = 3
    LinReg_Reset = DiagResult( _
        LR_Reset(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                 VarPtr(powers(0)), 2))
End Function

' Returns {DW statistic, rho (approx autocorrelation = 1 - DW/2), ""}
' p-value is not defined for Durbin-Watson; use the DW bounds table.
Public Function LinReg_DurbinWatson(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)

    Dim h As LongPtr
    h = LR_DurbinWatson(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p))
    If h = 0 Then
        LinReg_DurbinWatson = Array(GetLastErrorMsg())
        Exit Function
    End If
    Dim result(0 To 2) As Variant
    result(0) = LR_GetStatistic(h)
    result(1) = LR_GetAutocorrelation(h)
    result(2) = ""
    LR_Free h
    LinReg_DurbinWatson = result
End Function

' lagOrder: number of lags to test (default 1)
Public Function LinReg_BreuschGodfrey(yRange As Range, xRange As Range, _
                                       Optional lagOrder As Long = 1) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_BreuschGodfrey = DiagResult( _
        LR_BreuschGodfrey(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), lagOrder))
End Function

#End If  ' VBA7 (diagnostics section)

' ==============================================================================
' SECTION 7 - Prediction intervals wrapper
' ==============================================================================

' Compute OLS prediction intervals for new observations.
'
' Parameters:
'   yRange    - n training response values (single column/row)
'   xRange    - n x p training predictor matrix (no intercept column)
'   newXRange - n_new x p new predictor matrix (same columns as xRange)
'   alpha     - significance level (default 0.05 for 95% intervals)
'
' Returns an (n_new+1) x 4 Variant array:
'   Row 0:        header ["Predicted", "Lower", "Upper", "SE_Pred"]
'   Rows 1..n_new: one row per new observation
'
' On error returns a 1-element array containing the error message string.
#If VBA7 Then
Public Function LinReg_PredictionIntervals(yRange As Range, xRange As Range, _
                                            newXRange As Range, _
                                            Optional alpha As Double = 0.05) As Variant
    Dim y() As Double, X() As Double, newX() As Double
    Dim n As Long, p As Long, nNew As Long
    Dim h As LongPtr

    n    = yRange.Cells.Count
    p    = xRange.Columns.Count
    nNew = newXRange.Rows.Count

    y    = RangeToDoubleArray(yRange)
    X    = RangeToMatrix(xRange)
    newX = RangeToMatrix(newXRange)

    h = LR_PredictionIntervals(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                               VarPtr(newX(0)), CLng(nNew), alpha)
    If h = 0 Then
        LinReg_PredictionIntervals = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim pred()   As Double
    Dim lower()  As Double
    Dim upper()  As Double
    Dim sePred() As Double
    ReDim pred(0 To nNew - 1)
    ReDim lower(0 To nNew - 1)
    ReDim upper(0 To nNew - 1)
    ReDim sePred(0 To nNew - 1)

    LR_GetPredicted  h, VarPtr(pred(0)),   CLng(nNew)
    LR_GetLowerBound h, VarPtr(lower(0)),  CLng(nNew)
    LR_GetUpperBound h, VarPtr(upper(0)),  CLng(nNew)
    LR_GetSEPred     h, VarPtr(sePred(0)), CLng(nNew)

    LR_Free h

    Dim result() As Variant
    ReDim result(0 To nNew, 0 To 3)  ' +1 row for header

    result(0, 0) = "Predicted"
    result(0, 1) = "Lower"
    result(0, 2) = "Upper"
    result(0, 3) = "SE_Pred"

    Dim i As Long
    For i = 0 To nNew - 1
        result(i + 1, 0) = pred(i)
        result(i + 1, 1) = lower(i)
        result(i + 1, 2) = upper(i)
        result(i + 1, 3) = sePred(i)
    Next i

    LinReg_PredictionIntervals = result
End Function
#End If

' ==============================================================================
' SECTION 8 - WLS wrapper
' ==============================================================================

' Fit a Weighted Least Squares regression model.
'
' Parameters:
'   yRange  - n response values (single column or row)
'   xRange  - n rows x p predictor columns (no intercept)
'   wRange  - n non-negative observation weights (same layout as yRange)
'
' Returns the same (k+6) x 5 table as LinReg_OLS.
' On error returns a 1-element array with the error message string.
#If VBA7 Then
Public Function LinReg_WLS(yRange As Range, xRange As Range, wRange As Range) As Variant
    Dim y() As Double, X() As Double, w() As Double
    Dim n As Long, p As Long
    Dim h As LongPtr

    n = yRange.Cells.Count
    p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange)
    X = RangeToMatrix(xRange)
    w = RangeToDoubleArray(wRange)

    h = LR_WLS(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), VarPtr(w(0)))
    If h = 0 Then
        LinReg_WLS = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim k As Long
    k = LR_GetNumCoefficients(h)

    Dim coefs() As Double, ses() As Double
    Dim tstats() As Double, pvals() As Double
    ReDim coefs(0 To k - 1)
    ReDim ses(0 To k - 1)
    ReDim tstats(0 To k - 1)
    ReDim pvals(0 To k - 1)

    LR_GetCoefficients h, VarPtr(coefs(0)), k
    LR_GetStdErrors    h, VarPtr(ses(0)),   k
    LR_GetTStats       h, VarPtr(tstats(0)), k
    LR_GetPValues      h, VarPtr(pvals(0)),  k

    Dim r2 As Double, adjr2 As Double
    Dim fstat As Double, fp As Double, mse As Double
    r2    = LR_GetRSquared(h)
    adjr2 = LR_GetAdjRSquared(h)
    fstat = LR_GetFStatistic(h)
    fp    = LR_GetFPValue(h)
    mse   = LR_GetMSE(h)
    n     = LR_GetNumObservations(h)

    LR_Free h

    Dim nRows As Long
    nRows = k + 6
    Dim result() As Variant
    ReDim result(0 To nRows - 1, 0 To 4)

    result(0, 0) = "Term"
    result(0, 1) = "Coefficient"
    result(0, 2) = "Std Error"
    result(0, 3) = "t Stat"
    result(0, 4) = "p-Value"

    Dim i As Long
    For i = 0 To k - 1
        result(i + 1, 0) = IIf(i = 0, "Intercept", "X" & i)
        result(i + 1, 1) = coefs(i)
        result(i + 1, 2) = ses(i)
        result(i + 1, 3) = tstats(i)
        result(i + 1, 4) = pvals(i)
    Next i

    result(k + 2, 0) = "R-squared"
    result(k + 2, 1) = r2
    result(k + 3, 0) = "Adj R-squared"
    result(k + 3, 1) = adjr2
    result(k + 4, 0) = "F-stat"
    result(k + 4, 1) = fstat
    result(k + 4, 2) = "p(F)"
    result(k + 4, 3) = fp
    result(k + 5, 0) = "MSE"
    result(k + 5, 1) = mse
    result(k + 5, 2) = "n"
    result(k + 5, 3) = n

    LinReg_WLS = result
End Function
#End If

' ==============================================================================
' SECTION 9 - Influence diagnostic wrappers
' ==============================================================================
' VIF, Cook's Distance, DFFITS each return a simple 1-D column of values.
' DFBETAS returns a 2-D table (header row + n x p+1 matrix).
' ==============================================================================

#If VBA7 Then

' Return a (p x 1) array of VIF values, one per predictor (excluding intercept).
' Requires p >= 2 predictors.
Public Function LinReg_VIF(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)

    Dim h As LongPtr
    h = LR_VIF(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p))
    If h = 0 Then
        LinReg_VIF = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim nVals As Long
    nVals = LR_GetVectorLength(h)
    Dim vals() As Double
    ReDim vals(0 To nVals - 1)
    LR_GetVector h, VarPtr(vals(0)), nVals
    LR_Free h

    Dim result() As Variant
    ReDim result(0 To nVals - 1, 0 To 0)
    Dim i As Long
    For i = 0 To nVals - 1
        result(i, 0) = vals(i)
    Next i
    LinReg_VIF = result
End Function

' Return an (n x 1) array of Cook's distance values, one per observation.
Public Function LinReg_CooksDistance(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)

    Dim h As LongPtr
    h = LR_CooksDistance(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p))
    If h = 0 Then
        LinReg_CooksDistance = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim nVals As Long
    nVals = LR_GetVectorLength(h)
    Dim vals() As Double
    ReDim vals(0 To nVals - 1)
    LR_GetVector h, VarPtr(vals(0)), nVals
    LR_Free h

    Dim result() As Variant
    ReDim result(0 To nVals - 1, 0 To 0)
    Dim i As Long
    For i = 0 To nVals - 1
        result(i, 0) = vals(i)
    Next i
    LinReg_CooksDistance = result
End Function

' Return an (n x 1) array of DFFITS values, one per observation.
Public Function LinReg_DFFITS(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)

    Dim h As LongPtr
    h = LR_DFFITS(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p))
    If h = 0 Then
        LinReg_DFFITS = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim nVals As Long
    nVals = LR_GetVectorLength(h)
    Dim vals() As Double
    ReDim vals(0 To nVals - 1)
    LR_GetVector h, VarPtr(vals(0)), nVals
    LR_Free h

    Dim result() As Variant
    ReDim result(0 To nVals - 1, 0 To 0)
    Dim i As Long
    For i = 0 To nVals - 1
        result(i, 0) = vals(i)
    Next i
    LinReg_DFFITS = result
End Function

' Return an (n+1) x (p+1) table: header row then the DFBETAS matrix.
' Header row: "Obs", "Intercept", "X1", "X2", ...
' Each data row: observation index (1-based) then DFBETAS for each coefficient.
Public Function LinReg_DFBETAS(yRange As Range, xRange As Range) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)

    Dim h As LongPtr
    h = LR_DFBETAS(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p))
    If h = 0 Then
        LinReg_DFBETAS = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim nRows As Long, nCols As Long
    nRows = LR_GetMatrixRows(h)   ' = n observations
    nCols = LR_GetMatrixCols(h)   ' = p+1 parameters (intercept + slopes)

    Dim flat() As Double
    ReDim flat(0 To nRows * nCols - 1)
    LR_GetMatrix h, VarPtr(flat(0)), nRows * nCols
    LR_Free h

    ' Build result: header row + data rows
    Dim result() As Variant
    ReDim result(0 To nRows, 0 To nCols)  ' +1 row for header

    ' Header
    result(0, 0) = "Obs"
    result(0, 1) = "Intercept"
    Dim j As Long
    For j = 2 To nCols
        result(0, j) = "X" & (j - 1)
    Next j

    ' Data rows
    Dim i As Long
    For i = 0 To nRows - 1
        result(i + 1, 0) = i + 1  ' 1-based observation index
        For j = 0 To nCols - 1
            result(i + 1, j + 1) = flat(i * nCols + j)
        Next j
    Next i

    LinReg_DFBETAS = result
End Function

#End If  ' VBA7 (influence diagnostics)

' ==============================================================================
' SECTION 10 - Lambda path wrapper
' ==============================================================================

' Generate a glmnet-style lambda sequence.
'
' Parameters:
'   yRange          - response values
'   xRange          - predictor matrix (no intercept column)
'   nLambda         - number of lambda values (default 100)
'   lambdaMinRatio  - min/max lambda ratio (default 0.0 = auto)
'   alpha           - elastic-net mixing (0 = ridge, 1 = lasso; default 1.0)
'
' Returns an (L x 1) array of lambda values in descending order.
#If VBA7 Then
Public Function LinReg_LambdaPath(yRange As Range, xRange As Range, _
                                   Optional nLambda As Long = 100, _
                                   Optional lambdaMinRatio As Double = 0.0, _
                                   Optional alpha As Double = 1.0) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)

    Dim h As LongPtr
    h = LR_LambdaPath(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                      nLambda, lambdaMinRatio, alpha)
    If h = 0 Then
        LinReg_LambdaPath = Array(GetLastErrorMsg())
        Exit Function
    End If

    Dim nVals As Long
    nVals = LR_GetVectorLength(h)
    Dim vals() As Double
    ReDim vals(0 To nVals - 1)
    LR_GetVector h, VarPtr(vals(0)), nVals
    LR_Free h

    Dim result() As Variant
    ReDim result(0 To nVals - 1, 0 To 0)
    Dim i As Long
    For i = 0 To nVals - 1
        result(i, 0) = vals(i)
    Next i
    LinReg_LambdaPath = result
End Function
#End If

' ==============================================================================
' SECTION 11 - K-Fold cross validation wrappers
' ==============================================================================
' All four CV functions return a 1x6 array:
'   {nFolds, meanMSE, sdMSE, meanRMSE, sdRMSE, meanR2}
' On error: 1-element array with the error message string.
' ==============================================================================

#If VBA7 Then

' Helper: read scalar CV metrics from a handle into a 1x6 result.
Private Function CVResult(h As LongPtr) As Variant
    If h = 0 Then
        CVResult = Array(GetLastErrorMsg())
        Exit Function
    End If
    Dim result(0 To 5) As Variant
    result(0) = LR_GetCVNFolds(h)
    result(1) = LR_GetCVMeanMSE(h)
    result(2) = LR_GetCVStdMSE(h)
    result(3) = LR_GetCVMeanRMSE(h)
    result(4) = LR_GetCVStdRMSE(h)
    result(5) = LR_GetCVMeanR2(h)
    LR_Free h
    CVResult = result
End Function

' K-Fold CV for OLS.  nFolds defaults to 5.
Public Function LinReg_KFoldOLS(yRange As Range, xRange As Range, _
                                 Optional nFolds As Long = 5) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_KFoldOLS = CVResult( _
        LR_KFoldOLS(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), nFolds))
End Function

' K-Fold CV for Ridge.
Public Function LinReg_KFoldRidge(yRange As Range, xRange As Range, _
                                   lambda As Double, _
                                   Optional standardize As Boolean = True, _
                                   Optional nFolds As Long = 5) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_KFoldRidge = CVResult( _
        LR_KFoldRidge(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                      lambda, IIf(standardize, 1, 0), nFolds))
End Function

' K-Fold CV for Lasso.
Public Function LinReg_KFoldLasso(yRange As Range, xRange As Range, _
                                   lambda As Double, _
                                   Optional standardize As Boolean = True, _
                                   Optional nFolds As Long = 5) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_KFoldLasso = CVResult( _
        LR_KFoldLasso(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                      lambda, IIf(standardize, 1, 0), nFolds))
End Function

' K-Fold CV for Elastic Net.  alpha: 0 = Ridge, 1 = Lasso.
Public Function LinReg_KFoldElasticNet(yRange As Range, xRange As Range, _
                                        lambda As Double, alpha As Double, _
                                        Optional standardize As Boolean = True, _
                                        Optional nFolds As Long = 5) As Variant
    Dim y() As Double, X() As Double
    Dim n As Long, p As Long
    n = yRange.Cells.Count : p = xRange.Columns.Count
    y = RangeToDoubleArray(yRange) : X = RangeToMatrix(xRange)
    LinReg_KFoldElasticNet = CVResult( _
        LR_KFoldElasticNet(VarPtr(y(0)), CLng(n), VarPtr(X(0)), CLng(p), _
                           lambda, alpha, IIf(standardize, 1, 0), nFolds))
End Function

#End If  ' VBA7 (cross-validation section)

' ==============================================================================
' SECTION 12 - Utility
' ==============================================================================

' Return the linreg_core library version string (e.g. "0.8.0").
Public Function LinReg_Version() As String
    Dim buf(0 To 63) As Byte
    Dim written As Long
    written = LR_Version(VarPtr(buf(0)), 64)
    If written > 0 Then
        LinReg_Version = Left$(StrConv(buf, vbUnicode), written)
    Else
        LinReg_Version = ""
    End If
End Function
