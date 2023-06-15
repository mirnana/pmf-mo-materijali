<?php
    session_start();

    if(isset($_SESSION['zamisljeni_broj'])) {
        header("/zadatak3_pogodi.php");
        exit();
    }

    $_SESSION["zamisljeni_broj"] = rand(1, 100);
?>

<!DOCTYPE html>
<html lang="hr">

<head>
</head>

<body>
    <form action="zadatak3_pogodi.php" method="GET">
        <input type="text" name = "ime"> </input>
        <br>
        <input type="submit" value="Submit">
    </form>

    <br />


</body>

</html>