$path = ".\files\"
If(!(test-path -PathType container $path))
{
      New-Item -ItemType Directory -Path $path
}

Import-Csv ".\csv_file.csv" |
    ForEach-Object -Parallel {
        Invoke-WebRequest "https://infura-ipfs.io$($_.ref)" -O ".\files\$($_.id).mscz"
    } -ThrottleLimit 16


echo done