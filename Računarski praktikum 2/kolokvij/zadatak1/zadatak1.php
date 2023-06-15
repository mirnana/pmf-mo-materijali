<?php

session_start();

/* 
if(!isset($_SESSION['baza postavljena'])){
    $db = null; */
    try {
            $db = new PDO( "mysql:host=rp2.studenti.math.hr;dbname=kolokvij;charset=utf8", 'student', 'pass.mysql' );
            //$db-> setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        }
    catch( PDOException $e ) { exit( 'PDO Error: ' . $e->getMessage() ); }
    /* 
    $_SESSION['baza postavljena'] = 'da';
} */

function debug() {
        echo '<pre>';
        /* $st = $db->prepare("SELECT * FROM lokacije");
        $st->execute();
        while($row = $st->fetch()) {
            print_r($row[$_SESSION['smjer']]);
        }  */
        echo '$_POST = '; print_r( $_POST );
        echo '$_SESSION = '; print_r( $_SESSION );
        echo '</pre>';
    }

if(isset($_POST['lokacija'])) $_SESSION['lokacija'] = $_POST['lokacija'];


/* echo "proba:";
$st = $db->prepare("SELECT * FROM lokacije ");
$st->execute();
while($row = $st->fetch()) echo $row[$_SESSION['smjer']];
 */

if(isset($_POST['smjer'])) {    // kad odaberemo kuda hodati

    // prvo gledamo ima li tamo ičega
    $_SESSION['smjer'] = $_POST['smjer'];
    /* $st = $db->prepare("SELECT :smjer FROM lokacije WHERE naziv LIKE :lokacija");
    $st->execute(array('smjer' => $_SESSION['smjer'], 'lokacija' => $_SESSION['lokacija']));
    while($row = $st->fetch()) {
        if ($row[$_SESSION['smjer']] === '-') echo "Nije moguće ići u smjeru " . $_SESSION['smjer'] . ". Elf je ostao na lokaciji " . $_SESSION['lokacija'];
        else $_SESSION['lokacija'] = $row[$_SESSION['smjer']];  
    } */

    if($_SESSION['lokacije'][$_SESSION['lokacija']][$_SESSION['smjer']] === '-')
        echo "Nije moguće ići u smjeru " . $_SESSION['smjer'] . ". Elf je ostao na lokaciji " . $_SESSION['lokacija'];
    else $_SESSION['lokacija'] = $_SESSION['lokacije'][$_SESSION['lokacija']][$_SESSION['smjer']];

}

// uzimanje predmeta
foreach($_POST['odabir'] as $odabir) {
    $_SESSION['izgubljeni'][] = array($_SESSION['lokacija'] => $odabir);
    //$_SESSION['predmeti'][$_SESSION['lokacija']] // mičem s lokacije
}

debug();
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Naslov</title>

</head>
<body>

    <h2>Elf se nalazi na lokaciji: <?php echo $_SESSION['lokacija']; ?> </h2>

    <form method='post' action='zadatak1.php'>
        Odaberi smjer u kojem želiš ići:
        <select name="smjer">
            <option name="sjever" id="sjever">sjever</option>
            <option name="istok" id="istok">istok</option>
            <option name="zapad" id="zapad">zapad</option>
            <option name="jug" id="jug">jug</option>
        </select>
        <input type="submit" value="Idi!">
    </form>

    <form method='post' action='zadatak1.php'>
        Na lokaciji se nalaze predmeti:
        <table>
            <tr><th>Odabir</th><th>Predmet</th></tr>
        <?php
            $st = $db->prepare("SELECT * FROM predmeti WHERE lokacija LIKE :lokacija");
            $st->execute(array('lokacija' => $_SESSION['lokacija']));
            while($row = $st->fetch()) {
                ?>
                <tr>
                    <td>
                        <input type="checkbox" name="odabir[]" id="<?php echo $row['naziv']; ?>">
                    </td>
                    <td> 
                        <?php echo $row['naziv']; ?>
                    </td>
                </tr> 
                <?php
            }
        ?>
        </table>
        <input type="submit" value="Uzmi odabrane predmete!">
    </form>

</body>
</html>





<!-- note to self: na početku izvadit sve iz baze i spremit u session, pa onda samo drljat po arrayima z sessiona za sve funkcionalnosti