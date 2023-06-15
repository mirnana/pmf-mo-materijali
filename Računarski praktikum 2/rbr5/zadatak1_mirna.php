<?php
    function is_valid_hex($str) {
        if (preg_match("/^#[[:xdigit:]]{3}/#[[:xdigit:]]{6})$/", $str)) return TRUE;
        return FALSE;
    }

    function is_valid_name($str) {
        if(preg_match("^[a-z]{1,20}^", $str)) return TRUE;
        return FALSE;
    }

    if(isset($_POST["text_boja"]) && $_POST["text_boja"] != "") {
        if (preg_match("^#[[:xdigit:]]{6}^", $_POST["text_boja"])) $bg_color = $_POST["text_boja"];
        else {
            echo "unesi # i heksadecimalni sestoznamenkasti broj";
            $bg_color = "white";
        }
    }
    elseif(isset($_POST["boje"])) $bg_color = $_POST["boje"];
    elseif( isset($_COOKIE["bg_color"])) $bg_color = $_COOKIE["bg_color"];
    else $bg_color = "white";

    setcookie('bg_color', $bg_color, time() + 60*60);
?>

<!DOCTYPE html>
<html lang="hr">

<head>
    <style>
        body {
            background-color: <?php echo $bg_color ?>;
        }
    </style>
</head>

<body>

    <form action="<?php echo $_SERVER['PHP_SELF']?>" method="POST">

        <select name="boje" id="boje">
            <option value="yellow">Žuta</option>
            <option value="red">Crvena</option>
            <option value="blue">Plava</option>
            <option value="green">Zelena</option>
        </select>

        <br>

        <input type="text" name="text_boja">

        <input type="submit" value="Promijeni boju pozadine">
    </form>

    <?php
        //$boja = $_POST
    ?>

</body>

</html>