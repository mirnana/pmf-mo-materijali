<?php
    session_start();

    if (dobivamo_ime_igraca()) {
        $_SESSION['ime_igraca'] = $_GET['ime'];
    }

    if (!isset($_SESSION["ime_igraca"])) {
        exit("Trebate prvo posjetiti zadatak3_index.php");
    }

    $ime_igraca = $_SESSION['ime_igraca'];

    if (dobivamo_broj()) {
        if (rezultat() == 0) {    
            session_unset();
            session_destroy();
            echo "link";
        } else {
            echo "krivi broj";
        }
    }

    nacrtaj_formu_za_pogadjanje($ime_igraca);

    debug();
    exit();

    function dobivamo_ime_igraca() {
        return isset($_GET['ime']);
    }

    function dobivamo_broj() {
        return isset($_POST["broj"]);
    }

    function rezultat() {
        if($_SESSION["zamisljeni_broj"] === (int) $_POST["broj"]) {
            return 0;
        } elseif ($_SESSION["zamisljeni_broj"] < $_POST["broj"]) {
            return 1;
        } else {
            return  -1;
        }
    }

    function nacrtaj_formu_za_pogadjanje($ime_igraca) { ?>
        <!DOCTYPE html>
        <html lang="hr">
        
        <head>
        </head>
        
        <body>
            <h1> Bok, <?php echo $ime_igraca; ?> </h1>
        
            Pokusaj pogoditi zamisljeni broj:
        
            <form action="zadatak3_pogodi.php" method="POST">
                <input type="text" name="broj"> </input>
                <br>
                <input type="submit" value="Probaj pogoditi">
            </form>
        
            <br />        
        </body>
        
        </html>
        <?php
    }

    function debug() {
        echo '<pre>';
        echo '$_GET = ';
        print_r($_GET);
        echo '$_SESSION = ';
        print_r($_SESSION);
    }
?>
