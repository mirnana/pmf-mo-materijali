<?php

session_start();
inicijaliziraj();
obradi_klik();
crtaj_plocu_za_igru();
debug();

function crtaj_plocu_za_igru() {
    $na_redu = $_SESSION['na_redu'];
    echo '<h1>Na redu je: ' . player_name($na_redu) . "</h1>";
    echo '<form action="lel.php" method="post">';
    echo '<table>';
    for ($r = 0; $r < 3; ++$r) {
        echo '<tr>';
        for ($s = 0; $s < 3; ++$s) {
            echo '<td>';
            $this_player_name = player_name($_SESSION['ploca'][$r][$s]);
            $this_btn_name = btn_name($r, $s);
            echo '<input type="submit" name=' . $this_btn_name . ' value="' . $this_player_name . '">';
            echo '</td>';
        }
        echo '</tr>';
    }
    echo '</table>';
    echo '</form>';
}

function obradi_klik() {
    for ($r = 0; $r < 3; ++$r) {
        for ($s = 0; $s < 3; ++$s) {
            $this_btn_name = btn_name($r, $s);
            if (isset($_POST[$this_btn_name])) {
                obradi_klik_za($r, $s);
            }
        }
    }   
}

function obradi_klik_za($r, $s) {
    $na_redu = $_SESSION['na_redu'];
    if ($_SESSION['ploca'][$r][$s] !== 0) {
        exit("forbidden move");
    } else {
        $_SESSION['ploca'][$r][$s] = $na_redu;
        $_SESSION['na_redu'] = sljedeci_igrac($na_redu);    
    }
}

function btn_name($r, $s) {
    return 'btn_' . $r . '_' . $s;
}

function player_name($x) {
    if ($x == 0) {
        return "?";
    } elseif ($x == 1) {
        return "x";
    } else {
        return "o";
    }
}

function sljedeci_igrac($x) {
    if ($x == 1) {
        return 2;
    } elseif ($x == 2) {
        return 1;
    } else {
        exit("logic error");
    }
}

function inicijaliziraj() {
    if (!isset($_SESSION['na_redu'])) {
        $_SESSION['na_redu'] = 1;

        $_SESSION['ploca'] = [];

        for ($r = 0; $r < 3; ++$r) {
            $_SESSION['ploca'][$r] = [];
            for ($s = 0; $s < 3; ++$s) {
                $_SESSION['ploca'][$r][$s] = 0;
            }
        }
    }
}

function debug() {
    echo '<pre>';
    echo '$_GET = ';
    print_r($_GET);
    echo '$_POST = ';
    print_r($_POST);
    echo '$_SESSION = ';
    print_r($_SESSION);
    echo '</pre>';
}
?>