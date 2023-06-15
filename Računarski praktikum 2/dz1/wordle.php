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

<form action="wordle_igra.php" method="POST">
    Unesi svoje ime: 
    <input 
        type="text" 
        name = "ime"
        value="<?php if(isset($_SESSION['ime'])) echo $_SESSION['ime']; ?>"
        required> 
    </input>
    <br>
    Odaberi težinu igre:
    <select name="odabir_tezine" id="odabir_tezine">
        <option value="pet" selected>5 slova</option>
        <option value="sest">6 slova</option>
        <option value="sedam">7 slova</option>
    </select>
    <input type="submit" value="Započni igru!">
</form>
