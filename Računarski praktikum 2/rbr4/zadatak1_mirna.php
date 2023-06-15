<!DOCTYPE html>
<html lang="hr">
<title> Zadatak1 </title>
<head>
</head>

<body>
    <?php
        echo "<table style='border: 1px solid black;'>";
            for($i = 0; $i < 11; $i++) {
                echo "<tr style='border: 1px solid black;'>";
                for($j = 0; $j < 11; $j++) {
                    if( $i == 0 && $j == 0) echo "<th style='border: 1px solid black;'>*</th>";
                    else if ($i == 0) echo "<th style='border: 1px solid black;'>" . "$j" . "</th>";
                    else if ($j == 0) echo "<th style='border: 1px solid black;'>" . "$i" . "</th>";
                    else {
                        $umnozak = $i * $j;
                        echo "<td style='border: 1px solid black;'>" . "$umnozak" . "</th>";
                    }
                }
                echo "</tr>";
            }
        echo "</table>";
    ?>
</body>