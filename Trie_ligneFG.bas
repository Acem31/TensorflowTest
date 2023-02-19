Attribute VB_Name = "Module1"
Sub selectionCellulesEtTri()
    
    'D�terminer la derni�re ligne de la feuille
    Dim lastRow As Long
    lastRow = ActiveSheet.Cells(ActiveSheet.Rows.Count, "F").End(xlUp).Row

    'Parcourir chaque ligne de A1 � la derni�re ligne
    For i = 1 To lastRow
        'Trier la ligne de A � E par ordre croissant
        ActiveSheet.Range("F" & i & ":G" & i).Sort Key1:=ActiveSheet.Range("F" & i), Order1:=xlAscending, Header:=xlNo
    Next i
    
    'D�finir la plage de cellules � s�lectionner
    Dim plageCellules As Range
    Set plageCellules = ActiveSheet.Range("F1:G1")
    
    'Parcourir chaque ligne de A2 � la derni�re ligne
    For i = 2 To lastRow
        Set plageCellules = Union(plageCellules, ActiveSheet.Range("F" & i & ":G" & i))
    Next i
    
    'S�lectionner la plage de cellules
    plageCellules.Select

End Sub

