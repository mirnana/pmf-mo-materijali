<!DOCTYPE html>
<html lang="hr">
<title> Zadatak1 </title>
<head>
</head>

<body>
    <?php
        function my_sort(&$lista) {
            $broj_elemenata = count($lista);

            for($i = 0; $i < $broj_elemenata; $i++) {
                for($j = 0; $j < $broj_elemenata; $j++) {
                    if(strcmp($lista[$i], $lista[$j]) > 0) {
                        $tmp = $lista[$i];
                        $lista[$i] = $lista[$j];
                        $lista[$j] = $tmp;
                    }
                }
            }
        }


        $polje = [];
        for($i = 0; $i < 10; $i++) {
            $str = "";
            for($j = 0; $j < 5; $j++) {
                $slovo = chr(rand(ord('A'), ord('Z'))); //slučajan char između A i Z
                $str = $str . $slovo;
            }
            $polje[$i] = $str;
        }

        for($i = 0; $i < 10; $i++) {
            echo $polje[$i] . "\n";
        }

        echo "<br>";

        my_sort($polje);

        echo "sortirano polje: \n";
        for($i = 0; $i < 10; $i++) {
            echo $polje[$i] . "\n";
        }
    ?>
</body>