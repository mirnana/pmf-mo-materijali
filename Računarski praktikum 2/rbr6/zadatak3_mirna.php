<?php
$user = 'student'; 
$pass = 'pass.mysql';

try {
    $db = new PDO ('mysql:host=rp2.studenti.math.hr;dbname=imrovic;charset=utf8', $user, $pass);
    //$st = $db->query( 'SELECT JMBAG,Ime,Prezime FROM `Studenti 2`' );
    /*echo '<table>';
    while ($row = $st->fetch()) {
        echo '<tr>';
        echo '<td>';
        echo $row['JMBAG'];
        echo '</td>';

        echo '<td>';
        echo $row['Ime'];
        echo '</td>';

        echo '<td>';
        echo $row['Prezime'];
        echo '</td>';
        echo '</tr>';
    }
    echo '</table>';*/
   
    if(isset($_POST['ocjena'])){
        $ocjena = $_POST['ocjena'];
        $st = $db -> prepare("SELECT JMBAG, Ime, Prezime FROM `Studenti 2` WHERE ocjena = :ocjena ");
        $st -> execute(array('ocjena' => $ocjena));
        while($row = $st -> fetch()) echo "JMBAG je " . $row['JMBAG'] . "<br>" . 
                                            "Ime je " . $row['Ime'] . "<br>" . 
                                            "Prezime je " . $row['Prezime'] . "<br>";
    }
    $st2 = $db -> prepare ("INSERT INTO `Studenti 2` VALUES ( :test, :test2, :test3, :test4)");
    $st2 -> execute (array('test' => '9999999999', 'test2' => 'pero' , 'test3' => 'kvrzica', 'test4' => '6'));
} catch (PDOException $e) {
    echo "greška: " .  $e->getMessage(); 
    exit();
}


?>

<!DOCTYPE html>
<html lang="hr">

<head>
</head>

<body>
    Unesi ocjenu: 

    <form action="baza.php" method="POST">
        <input type="text" name="ocjena"> </input>
        <br>
        <input type="submit" value="Posalji">
    </form>

    <br />        
</body>

</html>