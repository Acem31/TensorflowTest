Attribute VB_Name = "Module1"
Sub selectionCellulesEtTri()
    
    'Déterminer la dernière ligne de la feuille
    Dim lastRow As Long
    lastRow = ActiveSheet.Cells(ActiveSheet.Rows.Count, "A").End(xlUp).Row

    'Parcourir chaque ligne de A1 à la dernière ligne
    For i = 1 To lastRow
        'Trier la ligne de A à E par ordre croissant
        ActiveSheet.Range("A" & i & ":E" & i).Sort Key1:=ActiveSheet.Range("A" & i), Order1:=xlAscending, Header:=xlNo
    Next i
    
    'Définir la plage de cellules à sélectionner
    Dim plageCellules As Range
    Set plageCellules = ActiveSheet.Range("A1:E1")
    
    'Parcourir chaque ligne de A2 à la dernière ligne
    For i = 2 To lastRow
        Set plageCellules = Union(plageCellules, ActiveSheet.Range("A" & i & ":E" & i))
    Next i
    
    'Sélectionner la plage de cellules
    plageCellules.Select

End Sub

