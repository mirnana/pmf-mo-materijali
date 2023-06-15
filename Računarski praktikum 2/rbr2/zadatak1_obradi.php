<?php
if (isset($_POST["input1"]) && isset($_POST["input2"])) {
    $prvi = (int) $_POST["input1"];
    $drugi = (int) $_POST["input2"];
    $zbroj = $prvi + $drugi;
}
?>

<!DOCTYPE html>
<html lang="hr">

<head>
</head>

<body>
    <form action="zadatak1_obradi.php" method="POST">
        <input type="text" name = "input1"> </input>
        <input type="text" name = "input2"> </input>
        <input type="submit" value="Submit">
    </form>

    <br />

    <?php
        if (isset($zbroj)) {
            echo $prvi;
            echo "<br />";
            echo $drugi;
            echo "<br />";
            echo "Zbroj: ";
            echo $zbroj;
        }
    ?>

</body>

</html>