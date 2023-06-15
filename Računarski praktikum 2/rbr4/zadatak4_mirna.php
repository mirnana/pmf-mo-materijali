<!DOCTYPE html>
<html lang="hr">
<title> Zadatak1 </title>
<head>
</head>

<body>
    <?php
        interface iUpravljiv {
            function idiRavno($x);
            function skreniDesno();
            function skreniLijevo();
        }

        class Vozilo implements iUpravljiv {
            private $ime = "Vozilo";
            private $smjer;
            private $koordinata_x;
            private $koordinata_y;
            private $prijedeni_km;

            function __construct($ime) {
                $this -> ime = $ime;
                $smjer = "N";
                $koordinata_x = 0;
                $koordinata_y = 0;
                $prijedeni_km = 0;
            }

            function gdje_sam() {
                echo "ime: " . $this -> ime . ", koordinate: (" . $this -> koordinata_x . ", " . $this -> koordinata_y . ")";
            }

            function idiRavno($x) {
                $this -> prijedeni_km += $x;
                if($smjer === "N") $this -> koordinata_x += $x;
                elseif($smjer === "S") $this -> koordinata_x += $x;
                elseif($smjer === "W") $this -> koordinata_y -= $x;
                elseif($smjer === "E") $this -> koordinata_y += $x;
            }

            function skreniDesno() {
                if($smjer === "N") $this -> smjer = "E";
                elseif($smjer === "E") $this -> smjer = "S";
                elseif($smjer === "S") $this -> smjer = "W";
                elseif($smjer === "W") $this -> smjer = "N";
            }

            function skreniLijevo() {
                if($smjer === "N") $smjer = "W";
                elseif($smjer === "E") $smjer = "N";
                elseif($smjer === "S") $smjer = "E";
                elseif($smjer === "W") $smjer = "S";
            }
        }

        class Auto extends Vozilo {
            private $potrosnja = 10;

            function __construct($potr) {
                parent::__construct("automobil");
                $this -> potrosnja = $potr;
            }

            function potroseniBenzin() {
                return $this -> prijedeni_km * $this -> potrosnja;
            }
        }

        class Tramvaj extends Vozilo {
            private $broj_linije;

            function __construct($br) {
                parent::_construct("tramvaj");
                $this -> broj_linije = $br;
            }

            function linija () {
                return $this -> broj_linije;
            }
        }

        


    ?>
</body>