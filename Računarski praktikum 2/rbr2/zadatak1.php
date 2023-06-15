<!DOCTYPE html>
<html lang="hr">

<head>
</head>

<body>

    <form action="zadatak1_obradi.php" method="get">
        <input type="text" name = "input1"> </input>
        <input type="text" name = "input2"> </input>
        <input type="submit" value="Submit">
    </form>

    <p>
        Zbroj: 

    <?php
        echo $_GET["input1"];
    ?>
    </p>

</body>

</html>