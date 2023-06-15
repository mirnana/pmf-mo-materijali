<?php 
session_start();
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8" />
	<title>Wordle!</title>
	<style></style>
</head>

<body>

    <h1>Wordle!</h1>

    <p>Čestitamo, <?php echo $_SESSION['ime']; ?>! Uspješno ste riješili igricu.</p>

    <form action='wordle.php'>
        <input type='submit' value='Igraj ponovno!'>
    </form>

    <?php
        unset($_SESSION['tezina']);
        unset($_SESSION['tocna_rijec']);
    ?>
</body>
</html>