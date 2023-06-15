<?php

session_start();

// definiramo bazene riječi za svaku od mogućih težina igrice:
$skup_rijeci_5 = array('konac', 'lonac', 'zvono', 'patka', 'letak', 'tetka', 'lakat', 'fakat');
$skup_rijeci_6 = array('slatka', 'glatka', 'vlatka', 'zvonce', 'jastuk', 'trokut', 'majica', 'knjiga');
$skup_rijeci_7 = array('neseser', 'mobitel', 'telefon', 'vratina', 'krapina', 'pasmina', 'virtuoz', 'teglica');

// prvi put kad se pozove ova skripta pri igranju igrice, inicijaliziramo elemente sessiona:
if(isset($_POST['odabir_tezine']) && isset($_POST['ime'])){
    $tezine = array('pet' => 5,
                'sest' => 6,
                'sedam' => 7            
    );
    $_SESSION['tezina'] = $tezine[$_POST['odabir_tezine']];

    $_SESSION['ime'] = $_POST['ime'];

    $_SESSION['broj_poteza'] = 0;

    $ime_skupa = 'skup_rijeci_' . $_SESSION['tezina'];
    $_SESSION['tocna_rijec'] = $$ime_skupa[array_rand($$ime_skupa)];

    $_SESSION['popis_pokusaja'] = array();

    $_SESSION['moguci_hintovi'] = str_split($_SESSION['tocna_rijec']);
    $_SESSION['hintovi'] = array();

    $_SESSION['moguci_veliki_hintovi'] = str_split($_SESSION['tocna_rijec']);
    $_SESSION['veliki_hintovi'] = array();
}

// obrađujemo situaciju kad korisnik želi upisati riječ:
if(isset($_POST['odabir_akcije'])) {
    if($_POST['odabir_akcije'] == 'pogodi' && isset($_POST['rijec'])) {
        $trenutna_rijec = $_POST['rijec'];
        //$trenutna_rijec = implode(str_word_count($_POST['rijec'], 1, 'čćđšžČĆĐŠŽ'));
        if(strlen($trenutna_rijec) != $_SESSION['tezina'])  // ovdje će strlen brojati nepravilno ako string sadrži č/ć/đ/š/ž
            echo "<script>alert('Neispravna duljina riječi!');</script>";
        elseif(strtolower($trenutna_rijec) == $_SESSION['tocna_rijec']) {
            header('Location: wordle_kraj.php');            // u slučaju pogotka slijedi čestitka i evt ponovna igra
            exit;
        }
        else array_push($_SESSION['popis_pokusaja'], $trenutna_rijec);

        $_SESSION['broj_poteza'] = $_SESSION['broj_poteza'] + 1;
    }
    // kad želi obični hint:
    elseif($_POST['odabir_akcije'] == 'hint') daj_hint();
    // i kad želi veliki hint:
    elseif($_POST['odabir_akcije'] == 'veliki_hint') daj_veliki_hint();
}

// debug();

// --------------------------------------------------------------------------------------------

function daj_hint() {
    $i = array_rand($_SESSION['moguci_hintovi']);
    array_push($_SESSION['hintovi'], $_SESSION['moguci_hintovi'][$i]);
    // kako bismo osigurali da se isti hint neće ponoviti dvaput, mičemo ga iz popisa:
    unset($_SESSION['moguci_hintovi'][$i]);
}

function daj_veliki_hint() {
    $i = array_rand($_SESSION['moguci_veliki_hintovi']);
    $_SESSION['veliki_hintovi'][$i] = $_SESSION['tocna_rijec'][$i];
    // kako bismo osigurali da se isti hint neće ponoviti dvaput, mičemo ga iz popisa:
    unset($_SESSION['moguci_veliki_hintovi'][$i]);
}

function prikazi_hintove() {
    echo "<table>";
    echo "<tr>";
    foreach($_SESSION['hintovi'] as $slovo)
        echo "<td style = 'background-color: #c9b458' >" . $slovo . "</td>";
        // #c9b458 je lijepa žuta, kao u pravom wordle-u
    echo "</tr>";
    echo "</table>";
}

function prikazi_velike_hintove() {
    echo "<table>";
    echo "<tr>";
    for($i = 0; $i < $_SESSION['tezina']; $i++) {
        echo "<td style = 'background-color: ";

        if(isset($_SESSION['veliki_hintovi'][$i])) 
            echo "#538d4e'>" . $_SESSION['tocna_rijec'][$i];    // #538d4e je lijepa zelena, kao u pravom wordle-u
        else 
            echo "#d8d8d8'>" . " ";                             // #d8d8d8 je lijepa siva, kao u pravom wordle-u
        echo "</td>";
    }
    echo "</tr>";
    echo "</table>";
}

function ispisi_pokusaje() {
    echo "<table>";
    foreach($_SESSION['popis_pokusaja'] as $pokusaj){
        $pokusaj = strtolower($pokusaj);    // u rječnicima imam samo lowercase riječi pa ovime 
                                            // izbjegavam probleme s usporedbom
        echo "<tr>";
        for($i = 0; $i < $_SESSION['tezina']; $i++) {
            echo "<td style = 'background-color: ";

            if($pokusaj[$i] == $_SESSION['tocna_rijec'][$i]) 
                echo "#538d4e";         // #538d4e je lijepa zelena, kao u pravom wordle-u
            elseif(in_array($pokusaj[$i], str_split($_SESSION['tocna_rijec'])))
                echo "#c9b458";         // #c9b458 je lijepa žuta, kao u pravom wordle-u
            else echo "#939598";        // #939598 je lijepa siva, kao u pravom wordle-u
            
            echo "' >" . $pokusaj[$i] . "</td>";
        }
        echo "</tr>";
    }
    echo "</table>";
}

function debug() {
    echo '<pre>';

    echo '$_POST = '; print_r( $_POST );
    echo '$_SESSION = '; print_r( $_SESSION );

    echo '</pre>';
}
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Wordle!</title>
	<style>
        table {
            border: none;
        }
        td {
            border: none;
            color: white;
            text-align: center;
            text-transform: uppercase;
            font-weight: bold;
            width: 30px;
            height: 30px;
        }
    </style>
</head>

<body>

    <h1>Wordle!</h1>

    Igrač: <?php echo $_SESSION['ime']; ?>
    <br>
    Težina: <?php echo $_SESSION['tezina'] . " slova"; ?>
    <br>
    Broj izvršenih poteza: <?php echo $_SESSION['broj_poteza']; ?>
    <br>
    Hintovi: <?php prikazi_hintove(); ?>
    <br>
    Veliki hintovi: <?php prikazi_velike_hintove(); ?>
    <br>
    Dosadašnji potezi: <?php ispisi_pokusaje(); ?>
    <br>

    <form action='wordle_igra.php' method='POST'>
        <input type='radio' name='odabir_akcije' value='hint'>Hint</input>
        <br>
        <input type='radio' name='odabir_akcije' value='veliki_hint'>Veliki hint</input>
        <br>
        <input type='radio' name='odabir_akcije' value='pogodi' checked>Probaj pogoditi riječ</input>
        <input type='text' name='rijec' id='unos_rijeci'/>
        <br>
        <input type="submit" value="Izvrši akciju!">
    </form>

</body>
</html>