<?php require_once __DIR__ . '/_header.php'; ?> 

<table>
    <tr>
        <th>Tko je posudio</th>
        <th>Naslov knjige</th>
        <th>Do kada je posuđena</th>
    </tr>

    <?php
        foreach( $loanList as $loan )
        {
            echo '<tr>';
            echo '<td>' . $loan->user()->name . ' ' . $loan->user()->surname . '</td>';
            echo '<td>' . $loan->book()->title . '</td>';
            echo '<td>' . $loan->lease_end . '</td>';
            echo '</tr>';
        }
    ?>
</table>

<?php require_once __DIR__ . '/_footer.php'; ?>
