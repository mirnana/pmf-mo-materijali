<?php
session_start();

try {
    $db = new PDO( "mysql:host=rp2.studenti.math.hr;dbname=kolokvij;charset=utf8", 'student', 'pass.mysql' );
    //$db-> setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
}
catch( PDOException $e ) { exit( 'PDO Error: ' . $e->getMessage() ); }

$_SESSION['lokacije'] = array();
$st = $db->prepare("SELECT * FROM lokacije");
$st->execute();
while($row = $st-> fetch()) {
    $_SESSION['lokacije'][$row['naziv']] = array(
        'sjever' => $row['sjever'],
        'istok' =̣> $row['istok'],
        'zapad' => $row['zapad'],
        'jug' =̣> $row['jug']
    );
}

$_SESSION['predmeti'] = array();
$st = $db->prepare("SELECT * FROM predmeti");
$st->execute();
while($row = $st-> fetch()) {
    $_SESSION['predmeti'][$row['lokacija']] = $row['naziv'];
}

$_SESSION['izgubljeni'] = array();

?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Naslov</title>

</head>
<body>

    <form action="zadatak1.php" method="post">
        Unesi naziv lokacije gdje se Elf nalazi:
        <input type="text" name="lokacija">
        <button type="submit">Kreni!</button>
    </form>

</body>
</html>