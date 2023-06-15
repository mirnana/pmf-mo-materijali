<?php 
    session_start();

    init();
    $pobjednik = obradi_klik();
    nacrtaj_plocu( $pobjednik );

    // debug();


    // --------------------------------
    function obradi_klik()
    {
        for( $r = 0; $r < 3; ++$r )
            for( $s = 0; $s < 3; ++$s )
            {
                $btn = 'btn_' . $r . $s;

                if( isset( $_POST[$btn] ) && $_SESSION['ploca'][$r][$s] === '?' )
                {
                    $_SESSION['ploca'][$r][$s] = $_SESSION['na_redu'];

                    if( $_SESSION['na_redu'] === 'x' )
                        $_SESSION['na_redu'] = 'o';
                    else
                        $_SESSION['na_redu'] = 'x';
                
                    return provjeri_pobjedu( $r, $s );
                }
            }

        return '?';
    }


    function provjeri_pobjedu( $r, $s )
    {
        if( $_SESSION['ploca'][$r][0] === $_SESSION['ploca'][$r][1] && 
            $_SESSION['ploca'][$r][0] === $_SESSION['ploca'][$r][2] )
        {
            return $_SESSION['ploca'][$r][0];   
        }

        if( $_SESSION['ploca'][0][$s] === $_SESSION['ploca'][1][$s] && 
            $_SESSION['ploca'][0][$s] === $_SESSION['ploca'][2][$s] )
        {
            return $_SESSION['ploca'][0][$s];   
        }

        if( $_SESSION['ploca'][0][0] === $_SESSION['ploca'][1][1] && 
            $_SESSION['ploca'][0][0] === $_SESSION['ploca'][2][2] )
        {
            return $_SESSION['ploca'][0][0];   
        }

        if( $_SESSION['ploca'][0][2] === $_SESSION['ploca'][1][1] && 
            $_SESSION['ploca'][0][2] === $_SESSION['ploca'][2][0] )
        {
            return $_SESSION['ploca'][1][1];   
        }

        return '?';
    }

    function nacrtaj_plocu( $pobjednik )
    {
        ?>
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Zadatak 4</title>
        </head>
        <body>
            Na redu je: <?php echo $_SESSION['na_redu']; ?>
            <br>
            <form action="zadatak4.php" method="post">
                <?php 
                    for( $r = 0; $r < 3; ++$r )
                    {
                        for( $s = 0; $s < 3; ++$s )
                        {
                            echo '<button ';

                            if( $_SESSION['ploca'][$r][$s] !== '?' )
                                echo 'disabled ';

                            echo 'type="submit" name="btn_' . $r . $s . '">';
                            echo $_SESSION['ploca'][$r][$s];
                            echo '</button>';
                        }
                        echo '<br>';
                        echo "\n";
                    }
                ?>
            </form>

            <?php 
                if( $pobjednik !== '?' )
                {
                    echo '<p>Pobjedio je igrač ' . $pobjednik . '!';

                    session_unset();
                    session_destroy();
                }
            ?>
        </body>
        </html>

        <?php
    }


    function debug()
    {
        echo '<pre>';

        echo '$_POST = '; print_r( $_POST );
        echo '$_SESSION = '; print_r( $_SESSION );

        echo '</pre>';
    }


    function init()
    {
        if( !isset( $_SESSION['na_redu'] ) )
        {
            $_SESSION['na_redu'] = 'x';
            
            $_SESSION['ploca'] = []; 
            for( $r = 0; $r < 3; ++$r )
            {
                $_SESSION['ploca'][$r] = [];

                for( $s = 0; $s < 3; ++$s )
                    $_SESSION['ploca'][$r][$s] = '?';
            }
        }
    }


?> 
